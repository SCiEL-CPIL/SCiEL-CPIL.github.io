import numpy as np
import time
import random

def construct():
        start_time = time.time()
        deadline = start_time + 85
        
        def count_nonzero(poly):
            return int(np.count_nonzero(poly))
            
        def evaluate(p):
            # p is a numpy array of length 41
            sq = np.convolve(p, p)
            return count_nonzero(sq)
            
        best_score = 25 # max possible for 12 terms is 23? 12+12-1 = 23.
        # Wait, 12 terms -> max 23 non-zero in square if no cancellation.
        # Baseline 1+x+...+x^11 -> 23 non-zero terms?
        # 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1.
        # All non-zero. So 23.
        
        best_poly = None
        
        def try_poly(p_list):
            nonlocal best_score, best_poly
            # p_list is list of 12 (index, value) pairs or a dense list?
            # Let's use dense list for consistency in storage
            p_dense = [0]*41
            for i, v in p_list:
                if 0 <= i < 41:
                    p_dense[i] = v
            
            # Check validity
            if len([x for x in p_dense if x != 0]) != 12: return
            if any(abs(x) > 64 for x in p_dense): return
            if p_dense[40] == 0 and max(i for i,v in p_list if v!=0) == 40: pass # Just check last coeff of P?
            # Task: "nonzero final coefficient" of the returned list.
            # The returned list is "coefficient list in ascending degree, length<=41".
            # It doesn't say the polynomial must have degree 40.
            # It says "length<=41".
            # "nonzero final coefficient" means the last element of the list is not 0.
            # So if we return [1, 0, 1], length 3, final coeff 1.
            # If we have a term at index 10, we can return length 11 list.
            # So we should trim trailing zeros.
            
            # Trim
            last_nonzero = 0
            for i in range(40, -1, -1):
                if p_dense[i] != 0:
                    last_nonzero = i
                    break
            
            if last_nonzero == -1: return
            
            p_trimmed = p_dense[:last_nonzero+1]
            
            # Check exactly 12 non-zero terms in p_trimmed
            if sum(1 for x in p_trimmed if x != 0) != 12: return
            
            score = evaluate(np.array(p_trimmed))
            
            if score < best_score:
                best_score = score
                best_poly = p_trimmed
                
        # Initialize with random
        def random_poly():
            indices = random.sample(range(41), 12)
            values = [random.choice([-1, 1, 2, -2, 3, -3]) for _ in range(12)]
            # Ensure last index in sorted order has non-zero value (it always does)
            # But we need the "final coefficient" of the returned list to be non-zero.
            # If we return trimmed list, the last coeff is the one at max index.
            # So we just need the term at max index to be non-zero.
            pairs = list(zip(sorted(indices), values))
            return pairs
            
        # Local search
        def local_search(pairs):
            curr_pairs = [p[:] for p in pairs]
            curr_score = evaluate(np.array([0]*41) ) # dummy
            # Build dense
            def get_dense(ps):
                d = [0]*41
                for i, v in ps:
                    d[i] = v
                return d
            curr_dense = get_dense(curr_pairs)
            curr_score = evaluate(np.array(curr_dense))
            
            for it in range(1000):
                if time.time() > deadline - 2:
                    break
                
                # Random move
                move_type = random.randint(0, 2)
                idx_in_list = random.randint(0, 11)
                old_i, old_v = curr_pairs[idx_in_list]
                
                if move_type == 0:
                    # Change value
                    new_v = random.choice([-1, 1, 2, -2, 3, -3, 4, -4, 5, -5])
                    if new_v == old_v: continue
                    curr_pairs[idx_in_list] = (old_i, new_v)
                elif move_type == 1:
                    # Move index
                    new_i = random.randint(0, 40)
                    # Check collision
                    if any(p[0] == new_i for p in curr_pairs if p[1] != 0 and p is not curr_pairs[idx_in_list]):
                         # Collision! We can't have two terms at same index in our representation of "12 terms"
                         # Unless we merge them? No, we maintain 12 distinct indices.
                         # So if new_i is occupied, we skip or swap.
                         # Let's just skip if occupied.
                         continue
                    curr_pairs[idx_in_list] = (new_i, old_v)
                else:
                    # Swap indices with another
                    j = random.randint(0, 11)
                    if j == idx_in_list: continue
                    i1, v1 = curr_pairs[idx_in_list]
                    i2, v2 = curr_pairs[j]
                    curr_pairs[idx_in_list] = (i2, v1)
                    curr_pairs[j] = (i1, v2)
                    
                # Evaluate
                new_dense = get_dense(curr_pairs)
                new_score = evaluate(np.array(new_dense))
                
                if new_score < curr_score:
                    curr_score = new_score
                    try_poly(curr_pairs)
                else:
                    # Revert
                    if move_type == 0:
                        curr_pairs[idx_in_list] = (old_i, old_v)
                    elif move_type == 1:
                        curr_pairs[idx_in_list] = (old_i, old_v)
                    else:
                        curr_pairs[idx_in_list] = (i1, v1)
                        curr_pairs[j] = (i2, v2)
                        
            return curr_pairs, curr_score

        # Main loop
        for _ in range(1000):
            if time.time() > deadline - 5:
                break
            start_pairs = random_poly()
            # Occasionally start from best
            if best_poly is not None and random.random() < 0.3:
                # Convert best_poly to pairs
                pairs = []
                for i, v in enumerate(best_poly):
                    if v != 0:
                        pairs.append((i, v))
                if len(pairs) == 12:
                    start_pairs = [p[:] for p in pairs]
                    
            res_pairs, res_score = local_search(start_pairs)
            
        # Final fallback
        if best_poly is None:
            best_poly = [1]*12
            
        return best_poly
