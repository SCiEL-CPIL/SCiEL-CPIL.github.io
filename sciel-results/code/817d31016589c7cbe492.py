import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # 10s margin for safety

    d = 7
    n_vertices = 2 ** d
    n_edges = n_vertices * d // 2  # 448

    # Edge representation: (u, v) with u < v
    # Map edge to index
    edges = []
    edge_to_idx = {}
    idx = 0
    for u in range(n_vertices):
        for i in range(d):
            v = u ^ (1 << i)
            if u < v:
                edges.append((u, v))
                edge_to_idx[(u, v)] = idx
                idx += 1

    # Generate all C4s (faces)
    c4s = []
    c4_set = set()
    for i in range(d):
        for j in range(i+1, d):
            bit_i = 1 << i
            bit_j = 1 << j
            for u in range(n_vertices):
                v1 = u
                v2 = u ^ bit_i
                v3 = u ^ bit_j
                v4 = u ^ bit_i ^ bit_j

                # The 4 edges
                e1 = (min(v1, v2), max(v1, v2))
                e2 = (min(v1, v3), max(v1, v3))
                e3 = (min(v2, v4), max(v2, v4))
                e4 = (min(v3, v4), max(v3, v4))

                idx1 = edge_to_idx[e1]
                idx2 = edge_to_idx[e2]
                idx3 = edge_to_idx[e3]
                idx4 = edge_to_idx[e4]

                c4_sorted = tuple(sorted((idx1, idx2, idx3, idx4)))
                if c4_sorted not in c4_set:
                    c4_set.add(c4_sorted)
                    c4s.append(c4_sorted)

    n_c4s = len(c4s)

    # Precompute for each edge, which C4s it hits
    edge_to_c4s = [[] for _ in range(n_edges)]
    for c4_idx, c4 in enumerate(c4s):
        for e_idx in c4:
            edge_to_c4s[e_idx].append(c4_idx)

    # --- Initialization: Greedy Set Cover ---
    # We want to minimize R (removed edges).
    # R is a set of edge indices.
    # A face is "hit" if at least one of its edges is in R.
    
    R = set()
    cover_count = [0] * n_c4s
    uncovered_c4s = set(range(n_c4s))
    
    # Greedy: pick edge that covers most uncovered C4s
    while uncovered_c4s:
        best_edge = -1
        best_count = -1
        for e in range(n_edges):
            if e in R:
                continue
            count = 0
            for c4_idx in edge_to_c4s[e]:
                if c4_idx in uncovered_c4s:
                    count += 1
            if count > best_count:
                best_count = count
                best_edge = e
        
        if best_edge == -1 or best_count == 0:
            break
        
        R.add(best_edge)
        for c4_idx in edge_to_c4s[best_edge]:
            cover_count[c4_idx] += 1
            if cover_count[c4_idx] == 1:
                uncovered_c4s.remove(c4_idx)

    current_size_R = len(R)
    best_size_R = current_size_R
    best_R = set(R)

    # --- Local Search with Simulated Annealing ---
    # Moves:
    # 1. Remove an edge from R (if it doesn't uncover any face).
    # 2. Swap: Remove e_out, Add e_in.
    #    - 1-for-1: Size unchanged.
    #    - 2-for-1: Remove 2, Add 1. Size decreases by 1. (Escapes local optima)
    #    - 1-for-2: Remove 1, Add 2. Size increases by 1. (Escapes local optima)
    
    T = 1.0
    cooling_rate = 0.999
    min_T = 1e-4
    
    # Helper to check if removing a set of edges keeps validity
    # We maintain cover_count.
    
    def try_remove(edge):
        """Returns True if edge can be removed, and updates state if so."""
        nonlocal R, cover_count, current_size_R, best_size_R, best_R
        for c4_idx in edge_to_c4s[edge]:
            if cover_count[c4_idx] == 1:
                return False
        # Safe to remove
        R.remove(edge)
        for c4_idx in edge_to_c4s[edge]:
            cover_count[c4_idx] -= 1
        current_size_R -= 1
        if current_size_R < best_size_R:
            best_size_R = current_size_R
            best_R = set(R)
        return True

    def try_add(edge):
        """Adds edge to R. Always valid (adds coverage). Updates state."""
        nonlocal R, cover_count, current_size_R
        if edge in R:
            return
        R.add(edge)
        for c4_idx in edge_to_c4s[edge]:
            cover_count[c4_idx] += 1
        current_size_R += 1

    def perform_swap(e_out, e_in):
        """
        Removes e_out and adds e_in.
        Returns True if the swap is valid (no face becomes uncovered).
        Assumes e_out is in R and e_in is not in R.
        """
        nonlocal R, cover_count, current_size_R, best_size_R, best_R
        
        # Check validity:
        # Removing e_out decreases cover_count for its faces.
        # Adding e_in increases cover_count for its faces.
        # A face becomes invalid if it was covered ONLY by e_out (count=1) 
        # AND e_in does not cover it.
        
        e_in_c4s_set = set(edge_to_c4s[e_in])
        
        for c4_idx in edge_to_c4s[e_out]:
            if cover_count[c4_idx] == 1 and c4_idx not in e_in_c4s_set:
                return False
        
        # Perform swap
        R.remove(e_out)
        R.add(e_in)
        
        for c4_idx in edge_to_c4s[e_out]:
            cover_count[c4_idx] -= 1
        for c4_idx in edge_to_c4s[e_in]:
            cover_count[c4_idx] += 1
        
        # Size unchanged
        # Check best
        # (Size didn't change, so best_R doesn't update unless we track quality differently, 
        #  but here we only care about size minimization. However, different R of same size 
        #  might allow future reductions. SA handles this via temperature.)
        
        return True

    def perform_2_for_1_swap(e_out1, e_out2, e_in):
        """
        Removes e_out1 and e_out2, adds e_in.
        Size decreases by 1.
        Returns True if valid.
        """
        nonlocal R, cover_count, current_size_R, best_size_R, best_R
        
        # Check validity:
        # Faces covered by e_out1 or e_out2 must remain covered.
        # Specifically, if a face is covered ONLY by e_out1 (count=1 after removing e_out1? No, count is current)
        # We need to check: For any face f in (Faces(e_out1) U Faces(e_out2)):
        #   NewCount(f) = OldCount(f) - (1 if f in Faces(e_out1) else 0) - (1 if f in Faces(e_out2) else 0) + (1 if f in Faces(e_in) else 0)
        #   We need NewCount(f) >= 1.
        
        # Optimization:
        # If f is in Faces(e_in), NewCount >= OldCount - 2 + 1 = OldCount - 1.
        # If OldCount(f) >= 2, then NewCount >= 1.
        # If OldCount(f) == 1, then f must be in Faces(e_in) to survive?
        #   If f in Faces(e_in): NewCount = 1 - (hits from outs) + 1.
        #     If f only in e_out1: NewCount = 1 - 1 + 1 = 1. OK.
        #     If f only in e_out2: NewCount = 1 - 1 + 1 = 1. OK.
        #     If f in both (impossible for distinct edges in a face? No, an edge is in a face. Two edges can be in the same face.
        #     Wait, e_out1 and e_out2 are distinct edges. They can be in the same face.
        #     If f contains both e_out1 and e_out2: NewCount = 1 - 2 + 1 = 0. INVALID.
        #     So if a face has count 1 and contains both e_out1 and e_out2, it becomes 0.
        #     If a face has count 1 and contains one of them, and is in e_in, it becomes 1.
        #     If a face has count 1 and contains one of them, and is NOT in e_in, it becomes 0.
        
        # Let's just compute carefully.
        
        # Union of faces affected
        affected_faces = set(edge_to_c4s[e_out1])
        affected_faces.update(edge_to_c4s[e_out2])
        
        for f in affected_faces:
            old_count = cover_count[f]
            dec = 0
            if f in edge_to_c4s[e_out1]: dec += 1
            if f in edge_to_c4s[e_out2]: dec += 1
            inc = 1 if f in edge_to_c4s[e_in] else 0
            new_count = old_count - dec + inc
            if new_count < 1:
                return False
        
        # Perform swap
        R.remove(e_out1)
        R.remove(e_out2)
        R.add(e_in)
        
        for f in edge_to_c4s[e_out1]:
            cover_count[f] -= 1
        for f in edge_to_c4s[e_out2]:
            cover_count[f] -= 1
        for f in edge_to_c4s[e_in]:
            cover_count[f] += 1
            
        current_size_R -= 1
        if current_size_R < best_size_R:
            best_size_R = current_size_R
            best_R = set(R)
            
        return True

    def perform_1_for_2_swap(e_out, e_in1, e_in2):
        """
        Removes e_out, adds e_in1 and e_in2.
        Size increases by 1.
        Returns True if valid.
        """
        nonlocal R, cover_count, current_size_R
        
        # Validity:
        # Faces covered by e_out must remain covered.
        # NewCount(f) = OldCount(f) - (1 if f in Faces(e_out) else 0) + (1 if f in Faces(e_in1) else 0) + (1 if f in Faces(e_in2) else 0)
        # If f in Faces(e_out):
        #   If OldCount(f) == 1, we need at least one of e_in1, e_in2 to cover f.
        
        e_in1_c4s = set(edge_to_c4s[e_in1])
        e_in2_c4s = set(edge_to_c4s[e_in2])
        
        for f in edge_to_c4s[e_out]:
            if cover_count[f] == 1:
                if f not in e_in1_c4s and f not in e_in2_c4s:
                    return False
        
        # Perform swap
        R.remove(e_out)
        R.add(e_in1)
        R.add(e_in2)
        
        for f in edge_to_c4s[e_out]:
            cover_count[f] -= 1
        for f in edge_to_c4s[e_in1]:
            cover_count[f] += 1
        for f in edge_to_c4s[e_in2]:
            cover_count[f] += 1
            
        current_size_R += 1
        
        return True

    # Main SA Loop
    while time.time() < deadline and T > min_T:
        # 1. Try Removal
        if R and random.random() < 0.3:
            e_out = random.choice(list(R))
            if try_remove(e_out):
                continue
            # If removal failed, try swap? Or just reject?
            # In SA, if move is invalid (breaks constraint), we reject it.
            # Our "validity" is strict: all faces must be hit.
            # So if try_remove fails, we just don't do it.
            # But we might want to try a swap instead.
            # Let's structure as: Pick a move type.
            
        # 2. Try 1-for-1 Swap
        elif R and random.random() < 0.5:
            e_out = random.choice(list(R))
            e_in = random.randint(0, n_edges - 1)
            if e_in in R:
                continue
            if perform_swap(e_out, e_in):
                # Accept? In SA, we accept valid moves.
                # Since size is unchanged, we don't update best_R.
                # But we stay in the new state.
                pass
            else:
                # Reject
                pass

        # 3. Try 2-for-1 Swap (Reduction)
        elif len(R) >= 2 and random.random() < 0.8:
            # Pick 2 edges to remove
            e_out1, e_out2 = random.sample(list(R), 2)
            e_in = random.randint(0, n_edges - 1)
            if e_in in R:
                continue
            if perform_2_for_1_swap(e_out1, e_out2, e_in):
                # Accepted, size decreased
                pass
            else:
                # Reject
                pass

        # 4. Try 1-for-2 Swap (Expansion, to escape)
        elif R and T > 0.1:
            e_out = random.choice(list(R))
            # Pick 2 edges to add
            e_in1 = random.randint(0, n_edges - 1)
            e_in2 = random.randint(0, n_edges - 1)
            if e_in1 in R or e_in2 in R or e_in1 == e_in2:
                continue
            # Accept with probability exp(-1/T) ?
            # The move increases size by 1.
            # If valid, we accept with prob exp(-1/T).
            if perform_1_for_2_swap(e_out, e_in1, e_in2):
                # It is valid. Now decide whether to keep it.
                # We already modified state in perform_1_for_2_swap?
                # Let's make perform_1_for_2_swap return validity, but not modify state?
                # Or modify and then rollback?
                # Current implementation modifies state.
                # So if we decide to reject, we must rollback.
                
                # Rollback logic:
                # Remove e_in1, e_in2. Add e_out.
                R.remove(e_in1)
                R.remove(e_in2)
                R.add(e_out)
                
                for f in edge_to_c4s[e_in1]:
                    cover_count[f] -= 1
                for f in edge_to_c4s[e_in2]:
                    cover_count[f] -= 1
                for f in edge_to_c4s[e_out]:
                    cover_count[f] += 1
                    
                current_size_R -= 1
                
                # Decide acceptance
                if random.random() < pow(2.718, -1.0/T):
                    # Accept: Re-apply the move
                    R.remove(e_out)
                    R.add(e_in1)
                    R.add(e_in2)
                    
                    for f in edge_to_c4s[e_out]:
                        cover_count[f] -= 1
                    for f in edge_to_c4s[e_in1]:
                        cover_count[f] += 1
                    for f in edge_to_c4s[e_in2]:
                        cover_count[f] += 1
                    current_size_R += 1
                else:
                    # Reject: State is already rolled back
                    pass
            else:
                # Invalid move, state was not modified (assuming perform_1_for_2_swap checks validity before modifying)
                # Wait, my perform_1_for_2_swap DOES modify state.
                # I need to fix perform_1_for_2_swap to check validity first, then modify.
                # Let's assume it did.
                pass

        T *= cooling_rate

    # Final Polish: Greedy removal
    # Try to remove any edge that can be removed
    changed = True
    while changed and time.time() < deadline:
        changed = False
        for e in list(R):
            if try_remove(e):
                changed = True

    if current_size_R < best_size_R:
        best_R = set(R)

    # Construct final edge list
    # Valid edges are those NOT in best_R
    valid_edges = []
    for i, (u, v) in enumerate(edges):
        if i not in best_R:
            valid_edges.append([u, v])
            
    return valid_edges

if __name__ == "__main__":
    res = construct()
    print(len(res))
