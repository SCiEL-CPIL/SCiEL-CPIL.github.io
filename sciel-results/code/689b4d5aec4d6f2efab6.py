import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin
    
    # Fallback: 67-vertex Paley
    def paley_tournament(p):
        squares = {x * x % p for x in range(1, p)}
        orientation = []
        for i in range(p):
            for j in range(i + 1, p):
                if (j - i) % p in squares:
                    orientation.append(0) # i -> j
                else:
                    orientation.append(1) # j -> i
        return {'n': p, 'orientation': orientation}

    best_result = paley_tournament(67)
    
    # Try to find smaller n
    for n in range(31, 40):
        if time.time() > deadline - 20.0:
            break
            
        # Local search for this n
        for restart in range(3):
            if time.time() > deadline:
                break
                
            # Initialize random tournament
            # N_plus[v] is an integer bitset
            N_plus = [0] * n
            for i in range(n):
                for j in range(i + 1, n):
                    if random.random() < 0.5:
                        N_plus[i] |= (1 << j)
                    else:
                        N_plus[j] |= (1 << i)
                        
            # Precompute 4-sets
            sets_4 = []
            masks_4 = []
            for i in range(n):
                for j in range(i + 1, n):
                    for k in range(j + 1, n):
                        for l in range(k + 1, n):
                            sets_4.append((i, j, k, l))
                            masks_4.append((1 << i) | (1 << j) | (1 << k) | (1 << l))
                            
            num_sets = len(sets_4)
            coverers = [[] for _ in range(num_sets)]
            
            # Initial coverage check
            uncovered_count = 0
            for idx, (s, mask) in enumerate(zip(sets_4, masks_4)):
                is_cov = False
                for v in range(n):
                    if (N_plus[v] & mask) == mask:
                        coverers[idx].append(v)
                        is_cov = True
                        break
                if not is_cov:
                    uncovered_count += 1
                    
            if uncovered_count == 0:
                # Build orientation
                orientation = []
                for i in range(n):
                    for j in range(i + 1, n):
                        if N_plus[i] & (1 << j):
                            orientation.append(0)
                        else:
                            orientation.append(1)
                best_result = {'n': n, 'orientation': orientation}
                return best_result
                
            # Local search
            flips = 0
            while time.time() < deadline - 10.0:
                if uncovered_count == 0:
                    break
                    
                # Pick a random uncovered set
                # We need a list of uncovered indices
                # For efficiency, we can just pick a random index and check if it's uncovered
                # But we need to know which are uncovered.
                # Maintain a set of uncovered indices?
                # Or just pick a random index, if it's covered, skip?
                # Better: maintain a list of uncovered indices.
                
                # Let's re-generate uncovered list every 100 flips?
                # Or update it incrementally.
                
                # Incremental update is better.
                # We need to know which sets are currently uncovered.
                # Let's maintain `uncovered_idx` as a set or list.
                
                # Actually, checking `if not coverers[idx]` is O(1).
                # So we can just pick a random idx.
                
                idx = random.randint(0, num_sets - 1)
                if coverers[idx]:
                    continue # Already covered
                
                s = sets_4[idx]
                mask = masks_4[idx]
                
                # Find a flip that might cover this set
                # Look for v in V \ S such that v covers S \ {u} for some u in S
                # i.e., v points to 3 of the 4 vertices in S.
                
                found_flip = False
                for v in range(n):
                    if v in s:
                        continue
                    # Check if v points to 3 of 4
                    # Count bits in N_plus[v] & mask
                    # If count is 3, then flipping the missing edge will cover S.
                    # But we need to know which edge to flip.
                    # Let common = N_plus[v] & mask
                    # missing = mask ^ common
                    # If popcount(common) == 3:
                    #   u = index of bit in missing
                    #   Flip edge (u, v) so that v -> u.
                    #   Currently u -> v (since v does not point to u).
                    #   So flip (u, v) to (v, u).
                    
                    common = N_plus[v] & mask
                    # Popcount
                    # Python 3.10+ has int.bit_count()
                    cnt = common.bit_count()
                    if cnt == 3:
                        missing = mask ^ common
                        u = missing.bit_length() - 1
                        # Flip edge between u and v
                        # Currently: u -> v (because v does not point to u, and tournament is complete)
                        # Wait, if v does not point to u, then u points to v.
                        # We want v to point to u.
                        # So remove u from N_plus[u]? No.
                        # N_plus[u] contains v. N_plus[v] does not contain u.
                        # We want N_plus[v] to contain u. N_plus[u] to not contain v.
                        
                        N_plus[v] |= (1 << u)
                        N_plus[u] &= ~(1 << v)
                        
                        # Update coverers for affected sets
                        # Affected sets: those containing u or v.
                        # We need to iterate through all sets and update?
                        # That is O(num_sets) which is 30k.
                        # Doing this every flip is too slow (30k * 1000 flips = 30M ops).
                        # We need incremental update.
                        
                        # Let's implement incremental update.
                        # This is complex to write inline.
                        # Given the time, maybe a simpler heuristic:
                        # Just flip and re-calculate global coverage every 50 flips?
                        # Global calc is 30k * 31 = 1M ops.
                        # 150s / 1M ops = 150 recalcs?
                        # If we recalc every 10 flips, we do 15 recalcs.
                        # This is feasible.
                        
                        found_flip = True
                        break
                        
                if not found_flip:
                    # Random flip
                    i = random.randint(0, n - 1)
                    j = random.randint(0, n - 1)
                    if i == j: continue
                    if N_plus[i] & (1 << j):
                        # i -> j, flip to j -> i
                        N_plus[i] &= ~(1 << j)
                        N_plus[j] |= (1 << i)
                    else:
                        # j -> i, flip to i -> j
                        N_plus[j] &= ~(1 << i)
                        N_plus[i] |= (1 << j)
                        
                flips += 1
                if flips % 20 == 0:
                    # Recalculate uncovered_count
                    uncovered_count = 0
                    for idx2 in range(num_sets):
                        if not coverers[idx2]:
                            uncovered_count += 1
                    # Update coverers? No, we need to update coverers to be accurate.
                    # If we don't update coverers, the "if coverers[idx]" check is wrong.
                    # So we must update coverers.
                    # Recalculating coverers from scratch is O(num_sets * n) = 1M.
                    # Let's do it.
                    coverers = [[] for _ in range(num_sets)]
                    for idx2, (s2, mask2) in enumerate(zip(sets_4, masks_4)):
                        for v in range(n):
                            if (N_plus[v] & mask2) == mask2:
                                coverers[idx2].append(v)
                                break
                        if not coverers[idx2]:
                            uncovered_count += 1
                            
            if uncovered_count == 0:
                orientation = []
                for i in range(n):
                    for j in range(i + 1, n):
                        if N_plus[i] & (1 << j):
                            orientation.append(0)
                        else:
                            orientation.append(1)
                best_result = {'n': n, 'orientation': orientation}
                return best_result
                
    return best_result
