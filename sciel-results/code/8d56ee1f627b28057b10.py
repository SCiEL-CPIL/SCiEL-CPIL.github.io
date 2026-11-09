import time
import random

def construct():
    start_time = time.time()
    deadline = 55.0 # Safety margin
    
    S = []
    # G[j] is a bitset of sums of subsets of S of size j
    # Max sum approx 512 * 48 = 24576. 1 << 24576 is huge but Python handles it.
    # We will use Python integers as bitsets.
    # G[0] = 1 (sum 0)
    G = [0] * 50
    G[0] = 1
    
    # D[a][j] is a bitset of sums of subsets of S\{a} of size j
    D = {}
    
    # Greedy construction
    for c in range(1, 513):
        if time.time() > deadline:
            break
            
        can_add = True
        
        # Check 1: c is not the mean of any subset of S
        # Subset size k >= 2
        # Sum must be k * c
        for k in range(2, len(S) + 1):
            target = k * c
            # Check if target is in G[k]
            if (G[k] >> target) & 1:
                can_add = False
                break
        
        if can_add:
            # Check 2: No a in S becomes the mean of a subset of S U {c}
            # For each a in S, and each subset size k >= 2 of S U {c} that includes c
            # Let j = k - 1. The subset from S\{a} has size j.
            # Sum of subset from S\{a} + c = k * a
            # Sum = k * a - c
            # We need to check if (k * a - c) is a sum of j distinct elements from S\{a}.
            # This is represented in D[a][j].
            for a in S:
                # k ranges from 2 to len(S) + 1
                # j = k - 1 ranges from 1 to len(S)
                for j in range(1, len(S) + 1):
                    k = j + 1
                    target = k * a - c
                    if target < 0:
                        continue
                    # Check if target is in D[a][j]
                    if (D[a][j] >> target) & 1:
                        can_add = False
                        break
                if not can_add:
                    break
        
        if can_add:
            # Add c to S
            # Save old G for D[c]
            G_old = G[:]
            
            # Update G with c
            # We iterate backwards to avoid using c multiple times
            for j in range(len(S) + 1, 0, -1):
                G[j] = G[j] | (G[j-1] << c)
                
            # Update D for existing a
            for a in S:
                for j in range(len(S) + 1, 0, -1):
                    D[a][j] = D[a][j] | (D[a][j-1] << c)
                    
            S.append(c)
            
            # Initialize D[c]
            # D[c][j] should be sums of subsets of S\{c} (which is old S) of size j
            # So D[c] = G_old
            D[c] = G_old[:]
            
            # If we reached 48, we can stop early
            if len(S) == 48:
                return S

    # If greedy didn't reach 48, try local search?
    # Given the speed, greedy might be sufficient.
    # But to be safe, if len(S) < 48, we can try swapping.
    # However, the bitset update for swaps is complex (removing an element).
    # Removing an element from bitsets is not straightforward (no division).
    # So we would have to rebuild the bitsets from scratch for a candidate swap.
    # Rebuild cost: O(N^2 * N) = O(N^3) bit ops.
    # N=48. 48^3 = 110k ops. Fast.
    
    if len(S) < 48:
        # Local search
        # Try to swap elements to increase size?
        # Actually, if greedy is stuck, it means no single addition works.
        # A swap (remove x, add y) might work.
        # But we want to INCREASE size.
        # If we are stuck at K < 48, we need to add an element.
        # If no single element can be added, we might need to remove one to make room for two?
        # Or just restart greedy with different order?
        
        # Let's try a few random restarts of greedy if we have time.
        attempts = 0
        best_S = S[:]
        while time.time() < deadline and attempts < 10:
            attempts += 1
            S_new = []
            G_new = [0] * 50
            G_new[0] = 1
            D_new = {}
            
            # Randomize start or order?
            # Just try adding in a slightly different order or skip some?
            # Actually, greedy on 1..512 is deterministic.
            # To vary, we can shuffle the candidates.
            candidates = list(range(1, 513))
            random.shuffle(candidates)
            
            for c in candidates:
                if time.time() > deadline:
                    break
                can_add = True
                for k in range(2, len(S_new) + 1):
                    target = k * c
                    if (G_new[k] >> target) & 1:
                        can_add = False
                        break
                if can_add:
                    for a in S_new:
                        for j in range(1, len(S_new) + 1):
                            k = j + 1
                            target = k * a - c
                            if target < 0: continue
                            if (D_new[a][j] >> target) & 1:
                                can_add = False
                                break
                        if not can_add: break
                
                if can_add:
                    G_old = G_new[:]
                    for j in range(len(S_new) + 1, 0, -1):
                        G_new[j] = G_new[j] | (G_new[j-1] << c)
                    for a in S_new:
                        for j in range(len(S_new) + 1, 0, -1):
                            D_new[a][j] = D_new[a][j] | (D_new[a][j-1] << c)
                    S_new.append(c)
                    D_new[c] = G_old[:]
                    
                    if len(S_new) > len(best_S):
                        best_S = S_new[:]
                        if len(best_S) == 48:
                            return best_S

        return best_S

    return S

