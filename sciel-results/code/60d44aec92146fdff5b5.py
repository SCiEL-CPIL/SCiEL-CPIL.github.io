"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 130  # Safety margin before 150s limit
    d = 7
    n = 2 ** d
    
    # Precompute all edges
    all_edges = []
    edge_to_idx = {}
    for x in range(n):
        for i in range(d):
            y = x ^ (1 << i)
            if x < y:
                idx = len(all_edges)
                all_edges.append((x, y))
                edge_to_idx[(x, y)] = idx
    
    m = len(all_edges)  # 448 edges
    
    # Precompute for each edge, the triples of other edges that form a C4 with it
    forbidden_triples = [[] for _ in range(m)]
    for idx, (x, y) in enumerate(all_edges):
        diff = x ^ y
        i = diff.bit_length() - 1
        for j in range(d):
            if j == i:
                continue
            xj = x ^ (1 << j)
            yj = y ^ (1 << j)
            e1 = (min(x, xj), max(x, xj))
            e2 = (min(y, yj), max(y, yj))
            e3 = (min(xj, yj), max(xj, yj))
            idx1 = edge_to_idx.get(e1)
            idx2 = edge_to_idx.get(e2)
            idx3 = edge_to_idx.get(e3)
            if idx1 is not None and idx2 is not None and idx3 is not None:
                forbidden_triples[idx].append((idx1, idx2, idx3))
    
    # Precompute all unique C4s as sorted tuples of 4 edge indices
    all_c4s = []
    seen_c4s = set()
    for idx in range(m):
        x, y = all_edges[idx]
        diff = x ^ y
        i = diff.bit_length() - 1
        for j in range(d):
            if j == i:
                continue
            xj = x ^ (1 << j)
            yj = y ^ (1 << j)
            e1 = (min(x, xj), max(x, xj))
            e2 = (min(y, yj), max(y, yj))
            e3 = (min(xj, yj), max(xj, yj))
            idx1 = edge_to_idx.get(e1)
            idx2 = edge_to_idx.get(e2)
            idx3 = edge_to_idx.get(e3)
            if idx1 is not None and idx2 is not None and idx3 is not None:
                c4 = tuple(sorted([idx, idx1, idx2, idx3]))
                if c4 not in seen_c4s:
                    seen_c4s.add(c4)
                    all_c4s.append(c4)
    
    # For each edge, list of C4s it belongs to
    edge_to_c4s = [[] for _ in range(m)]
    for c4_idx, c4 in enumerate(all_c4s):
        for e in c4:
            edge_to_c4s[e].append(c4_idx)
    
    def can_add(idx, chosen):
        """Check if edge idx can be added without creating a C4."""
        for (i1, i2, i3) in forbidden_triples[idx]:
            if chosen[i1] and chosen[i2] and chosen[i3]:
                return False
        return True
    
    def greedy_polish(chosen, count):
        """Try to add all possible edges in multiple passes."""
        improved = True
        while improved:
            improved = False
            for idx in range(m):
                if not chosen[idx] and can_add(idx, chosen):
                    chosen[idx] = True
                    count += 1
                    improved = True
        return count
    
    def mcs_greedy(seed):
        """
        Maximum Cardinality Search heuristic:
        Maintain blocked_count for each unselected edge.
        Pick edge with minimum blocked_count (least constrained).
        """
        rng = random.Random(seed)
        chosen = np.zeros(m, dtype=bool)
        count = 0
        
        # c4_selected_count[c4_idx] = number of selected edges in this C4
        c4_selected_count = np.zeros(len(all_c4s), dtype=np.int32)
        
        # blocked_count[i] = number of C4s containing i where the other 3 are all selected
        blocked_count = np.zeros(m, dtype=np.int32)
        
        unselected = set(range(m))
        
        while unselected:
            if time.time() > deadline - 5:
                break
            
            # Find edge with minimum blocked_count
            best_idx = -1
            best_bc = float('inf')
            # Sample a subset for speed, or use full scan if small
            if len(unselected) <= 100:
                candidates = list(unselected)
            else:
                candidates = random.sample(list(unselected), 100)
            
            for idx in candidates:
                bc = blocked_count[idx]
                if bc < best_bc:
                    best_bc = bc
                    best_idx = idx
            
            if best_idx == -1 or best_bc > 0:
                # If no edge with blocked_count 0, try to find one
                for idx in unselected:
                    if blocked_count[idx] == 0:
                        best_idx = idx
                        best_bc = 0
                        break
                if best_idx == -1:
                    break
            
            unselected.discard(best_idx)
            chosen[best_idx] = True
            count += 1
            
            # Update blocked_count for all edges in C4s that best_idx belongs to
            for c4_idx in edge_to_c4s[best_idx]:
                c4 = all_c4s[c4_idx]
                c4_selected_count[c4_idx] += 1
                new_count = c4_selected_count[c4_idx]
                
                if new_count == 3:
                    # One edge in this C4 is now blocked
                    for e in c4:
                        if not chosen[e]:
                            blocked_count[e] += 1
        
        return chosen, count
    
    def local_search(chosen, count, T_init=2.0, T_min=0.01, cooling=0.99999, max_time=None):
        if max_time is None:
            max_time = deadline - time.time() - 5
        
        best_count = count
        best_chosen = chosen.copy()
        no_improve = 0
        max_no_improve = 50000
        
        T = T_init
        while time.time() < max_time and T > T_min:
            idx = random.randint(0, m - 1)
            
            if not chosen[idx]:
                if can_add(idx, chosen):
                    chosen[idx] = True
                    count += 1
                    if count > best_count:
                        best_count = count
                        best_chosen = chosen.copy()
                        no_improve = 0
                    else:
                        no_improve += 1
                else:
                    # Try 2-opt swap: remove a conflicting edge, add idx
                    conflicts = []
                    for (i1, i2, i3) in forbidden_triples[idx]:
                        if chosen[i1] and chosen[i2] and chosen[i3]:
                            conflicts.extend([i1, i2, i3])
                    
                    if conflicts:
                        random.shuffle(conflicts)
                        success = False
                        for remove_idx in conflicts[:5]:
                            if not chosen[remove_idx]:
                                continue
                            chosen[remove_idx] = False
                            count -= 1
                            if can_add(idx, chosen):
                                chosen[idx] = True
                                count += 1
                                success = True
                                if count > best_count:
                                    best_count = count
                                    best_chosen = chosen.copy()
                                    no_improve = 0
                                else:
                                    no_improve += 1
                                break
                            else:
                                chosen[remove_idx] = True
                                count += 1
                        if not success:
                            no_improve += 1
                    else:
                        no_improve += 1
            else:
                # Consider removing idx
                chosen[idx] = False
                count -= 1
                added = 0
                added_edges = []
                for other in range(m):
                    if not chosen[other] and can_add(other, chosen):
                        chosen[other] = True
                        count += 1
                        added += 1
                        added_edges.append(other)
                
                if added > 1:
                    if count > best_count:
                        best_count = count
                        best_chosen = chosen.copy()
                        no_improve = 0
                    else:
                        no_improve += 1
                else:
                    for other in added_edges:
                        chosen[other] = False
                        count -= 1
                    chosen[idx] = True
                    count += 1
                    no_improve += 1
            
            T *= cooling
            
            if no_improve > max_no_improve:
                chosen = best_chosen.copy()
                count = best_count
                T = max(T, 1.5)
                no_improve = 0
        
        return best_chosen, best_count
    
    # Multi-round search with different initializations
    best_count = 0
    best_chosen = None
    
    # Round 1: MCS initialization
    for round_num in range(8):
        if time.time() > deadline - 15:
            break
        
        if round_num == 0:
            # MCS with seed 42
            chosen, count = mcs_greedy(42)
        elif round_num == 1:
            # MCS with seed 123
            chosen, count = mcs_greedy(123)
        elif round_num == 2:
            # MCS with seed 456
            chosen, count = mcs_greedy(456)
        elif round_num == 3:
            # Start with all edges, then remove to fix conflicts
            chosen = np.ones(m, dtype=bool)
            count = m
            violated = []
            for idx in range(m):
                for (i1, i2, i3) in forbidden_triples[idx]:
                    if chosen[idx] and chosen[i1] and chosen[i2] and chosen[i3]:
                        violated.append((idx, i1, i2, i3))
            
            random.seed(42)
            random.shuffle(violated)
            for (a, b, c, e) in violated:
                if chosen[a] and chosen[b] and chosen[c] and chosen[e]:
                    to_remove = random.choice([a, b, c, e])
                    chosen[to_remove] = False
                    count -= 1
            
            count = greedy_polish(chosen, count)
        elif round_num == 4:
            # Random initialization with density 0.5
            random.seed(789)
            chosen = np.zeros(m, dtype=bool)
            for idx in range(m):
                if random.random() < 0.5:
                    if can_add(idx, chosen):
                        chosen[idx] = True
            count = chosen.sum()
            count = greedy_polish(chosen, count)
        elif round_num == 5:
            # Random initialization with density 0.4
            random.seed(321)
            chosen = np.zeros(m, dtype=bool)
            for idx in range(m):
                if random.random() < 0.4:
                    if can_add(idx, chosen):
                        chosen[idx] = True
            count = chosen.sum()
            count = greedy_polish(chosen, count)
        elif round_num == 6:
            # MCS with seed 789
            chosen, count = mcs_greedy(789)
        else:
            # MCS with seed 1000
            chosen, count = mcs_greedy(1000)
        
        if best_count == 0 or count > best_count:
            best_count = count
            best_chosen = chosen.copy()
        
        # Local search
        remaining = deadline - time.time() - 10
        if remaining > 5:
            chosen, count = local_search(chosen, count, T_init=2.0, T_min=0.01, cooling=0.99999, max_time=remaining)
            count = greedy_polish(chosen, count)
            if count > best_count:
                best_count = count
                best_chosen = chosen.copy()
    
    # Final polish
    count = greedy_polish(best_chosen, best_count)
    if count > best_count:
        best_count = count
    
    # Convert to output
    out = []
    for idx in range(m):
        if best_chosen[idx]:
            x, y = all_edges[idx]
            out.append([x, y])
    
    return out
# EVOLVE-BLOCK-END
