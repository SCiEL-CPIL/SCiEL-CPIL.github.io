import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 55.0
    n = 256
    best_A = []
    best_B = []
    best_score = 0
    rng = random.Random(42)

    # Helper to check if x can be added to set S
    # S: list of elements
    # S_sums: set of sums
    # S_diffs: set of diffs
    # other_diffs: set of diffs of the OTHER set
    def can_add(x, S, S_sums, S_diffs, other_diffs):
        # 1. Check Sidon condition for S ∪ {x}
        # New sums are x+x and x+a for all a in S
        if x + x in S_sums:
            return False
        for a in S:
            if x + a in S_sums:
                return False
        
        # 2. Check difference disjointness
        # New diffs are x-a and a-x for all a in S
        for a in S:
            if (x - a) in other_diffs:
                return False
            if (a - x) in other_diffs:
                return False
        
        return True

    def get_diffs(S):
        diffs = set()
        for i in range(len(S)):
            for j in range(len(S)):
                if i != j:
                    diffs.add(S[i] - S[j])
        return diffs

    def get_sums(S):
        sums = set()
        for i in range(len(S)):
            for j in range(i, len(S)):
                sums.add(S[i] + S[j])
        return sums

    # Greedy construction with random restarts
    for trial in range(2000):
        if time.time() > deadline:
            break
        
        elements = list(range(1, n + 1))
        rng.shuffle(elements)
        
        A = []
        B = []
        sums_A = set()
        sums_B = set()
        diffs_A = set()
        diffs_B = set()
        
        for x in elements:
            if time.time() > deadline:
                break
            
            can_A = can_add(x, A, sums_A, diffs_A, diffs_B)
            can_B = can_add(x, B, sums_B, diffs_B, diffs_A)
            
            if can_A and can_B:
                # Balance: add to the smaller set
                if len(A) <= len(B):
                    A.append(x)
                    # Update sums_A
                    for a in A[:-1]:
                        sums_A.add(x + a)
                    sums_A.add(x + x)
                    # Update diffs_A
                    for a in A[:-1]:
                        diffs_A.add(x - a)
                        diffs_A.add(a - x)
                else:
                    B.append(x)
                    for b in B[:-1]:
                        sums_B.add(x + b)
                    sums_B.add(x + x)
                    for b in B[:-1]:
                        diffs_B.add(x - b)
                        diffs_B.add(b - x)
            elif can_A:
                A.append(x)
                for a in A[:-1]:
                    sums_A.add(x + a)
                sums_A.add(x + x)
                for a in A[:-1]:
                    diffs_A.add(x - a)
                    diffs_A.add(a - x)
            elif can_B:
                B.append(x)
                for b in B[:-1]:
                    sums_B.add(x + b)
                sums_B.add(x + x)
                for b in B[:-1]:
                    diffs_B.add(x - b)
                    diffs_B.add(b - x)
        
        score = len(A) + len(B)
        if score > best_score:
            best_A = A[:]
            best_B = B[:]
            best_score = score
        
        # Local Search: Try to improve the current solution
        # We recompute sets from scratch for safety in local search steps
        improved = True
        local_deadline = min(deadline, start_time + 55.0)
        
        while improved and time.time() < local_deadline:
            improved = False
            
            # Recompute auxiliary sets
            sums_A = get_sums(A)
            sums_B = get_sums(B)
            diffs_A = get_diffs(A)
            diffs_B = get_diffs(B)
            
            # 1. Try to add unassigned elements
            assigned = set(A) | set(B)
            unassigned = [x for x in range(1, n + 1) if x not in assigned]
            
            for x in unassigned:
                if time.time() > local_deadline:
                    break
                if can_add(x, A, sums_A, diffs_A, diffs_B):
                    A.append(x)
                    improved = True
                    break
                if can_add(x, B, sums_B, diffs_B, diffs_A):
                    B.append(x)
                    improved = True
                    break
            
            if improved:
                continue
            
            # 2. Try swaps: move x from A to B, or B to A
            # This can help if one set is "full" but the other has room,
            # or if moving an element frees up difference space.
            
            # Try moving from A to B
            for i, x in enumerate(A):
                if time.time() > local_deadline:
                    break
                A_new = A[:i] + A[i+1:]
                B_new = B + [x]
                
                # Check if A_new is Sidon
                if len(A_new) > 1:
                    s_new = get_sums(A_new)
                    if len(s_new) != len(A_new)*(len(A_new)+1)//2:
                        continue
                else:
                    s_new = get_sums(A_new)
                
                # Check if B_new is Sidon
                s_b_new = get_sums(B_new)
                if len(s_b_new) != len(B_new)*(len(B_new)+1)//2:
                    continue
                
                # Check diff disjointness
                d_a_new = get_diffs(A_new)
                d_b_new = get_diffs(B_new)
                if not (d_a_new & d_b_new):
                    # Valid move. Does it improve score?
                    # Score is len(A)+len(B), which is constant here.
                    # But it might allow further additions.
                    # We accept it to explore state space.
                    # To avoid infinite loops, we only accept if it leads to a better "potential"
                    # or we just accept and rely on the "add unassigned" step.
                    # Let's accept if it balances better or just accept randomly to escape local optima.
                    # Simple heuristic: accept if it reduces the max(len(A), len(B))
                    if max(len(A_new), len(B_new)) < max(len(A), len(B)) or (max(len(A_new), len(B_new)) == max(len(A), len(B)) and rng.random() < 0.1):
                        A = A_new
                        B = B_new
                        improved = True
                        break
            
            if improved:
                continue
            
            # Try moving from B to A
            for i, x in enumerate(B):
                if time.time() > local_deadline:
                    break
                B_new = B[:i] + B[i+1:]
                A_new = A + [x]
                
                s_a_new = get_sums(A_new)
                if len(s_a_new) != len(A_new)*(len(A_new)+1)//2:
                    continue
                
                s_b_new = get_sums(B_new)
                if len(B_new) > 1:
                    if len(s_b_new) != len(B_new)*(len(B_new)+1)//2:
                        continue
                else:
                    s_b_new = get_sums(B_new)
                
                d_a_new = get_diffs(A_new)
                d_b_new = get_diffs(B_new)
                if not (d_a_new & d_b_new):
                    if max(len(A_new), len(B_new)) < max(len(A), len(B)) or (max(len(A_new), len(B_new)) == max(len(A), len(B)) and rng.random() < 0.1):
                        A = A_new
                        B = B_new
                        improved = True
                        break
            
            if improved:
                continue

            # 3. Try swaps between A and B
            # Swap x in A with y in B
            for i, x in enumerate(A):
                if time.time() > local_deadline:
                    break
                for j, y in enumerate(B):
                    A_new = A[:i] + A[i+1:] + [y]
                    B_new = B[:j] + B[j+1:] + [x]
                    
                    s_a_new = get_sums(A_new)
                    if len(s_a_new) != len(A_new)*(len(A_new)+1)//2:
                        continue
                    
                    s_b_new = get_sums(B_new)
                    if len(s_b_new) != len(B_new)*(len(B_new)+1)//2:
                        continue
                    
                    d_a_new = get_diffs(A_new)
                    d_b_new = get_diffs(B_new)
                    if not (d_a_new & d_b_new):
                        # Accept swap if it improves balance or randomly
                        if max(len(A_new), len(B_new)) < max(len(A), len(B)) or (max(len(A_new), len(B_new)) == max(len(A), len(B)) and rng.random() < 0.05):
                            A = A_new
                            B = B_new
                            improved = True
                            break
                if improved:
                    break
            
            if improved:
                continue

            # 4. Try swapping an element in A with an unassigned element
            # This is effectively "remove and add", which is covered by "add unassigned"
            # if the removal makes room. But "add unassigned" doesn't remove.
            # So we need explicit swap with unassigned.
            
            # Try swapping x in A with z in unassigned
            for i, x in enumerate(A):
                if time.time() > local_deadline:
                    break
                for z in unassigned:
                    A_new = A[:i] + A[i+1:] + [z]
                    
                    s_a_new = get_sums(A_new)
                    if len(s_a_new) != len(A_new)*(len(A_new)+1)//2:
                        continue
                    
                    d_a_new = get_diffs(A_new)
                    # diffs_B is unchanged
                    if not (d_a_new & diffs_B):
                        # Accept if it allows adding more later?
                        # Just accept to explore.
                        if rng.random() < 0.1:
                            A = A_new
                            improved = True
                            break
                if improved:
                    break
            
            if improved:
                continue

            # Try swapping y in B with z in unassigned
            for j, y in enumerate(B):
                if time.time() > local_deadline:
                    break
                for z in unassigned:
                    B_new = B[:j] + B[j+1:] + [z]
                    
                    s_b_new = get_sums(B_new)
                    if len(s_b_new) != len(B_new)*(len(B_new)+1)//2:
                        continue
                    
                    d_b_new = get_diffs(B_new)
                    if not (d_b_new & diffs_A):
                        if rng.random() < 0.1:
                            B = B_new
                            improved = True
                            break
                if improved:
                    break
        
        # Update best
        score = len(A) + len(B)
        if score > best_score:
            best_A = A[:]
            best_B = B[:]
            best_score = score

    # Ensure nonempty
    if not best_A:
        best_A = [1]
    if not best_B:
        best_B = [2]
    
    return [best_A, best_B]
