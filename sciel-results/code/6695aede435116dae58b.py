import time
import random

N = 256

def construct():
    start_time = time.time()
    deadline = start_time + 25.0
    
    best_size = N + 1
    best_set = []
    
    for iter in range(500): # More iterations possible with speedup
        if time.time() > deadline:
            break
            
        random.seed(iter)
        
        S_mask = 0
        sums_mask = 0
        available_mask = ((1 << (N + 1)) - 1) ^ 1
        
        S_list = []
        
        while available_mask:
            # Get list of available elements
            avail_list = []
            temp_mask = available_mask
            while temp_mask:
                lsb = temp_mask & -temp_mask
                z = lsb.bit_length() - 1
                avail_list.append(z)
                temp_mask ^= lsb
            
            if not avail_list:
                break

            random.shuffle(avail_list)
            
            best_z = -1
            max_score = -1
            
            # Pre-extract S elements for faster access in inner loop
            S_elems = []
            temp_S = S_mask
            while temp_S:
                lsb_S = temp_S & -temp_S
                a = lsb_S.bit_length() - 1
                S_elems.append(a)
                temp_S ^= lsb_S
            
            for z in avail_list:
                # Check if z is valid to add
                # 1. 2z not in sums
                if (sums_mask >> (2*z)) & 1:
                    continue
                # 2. No collision with existing sums
                if (sums_mask >> z) & S_mask:
                    continue
                
                # Calculate blocked count efficiently
                # w is blocked if:
                # 1. 2w = z + a  => w = (z+a)/2
                # 2. w + a' = 2z => w = 2z - a'
                # 3. w + a' = z + a => w = z + a - a'
                
                blocked_candidates = set()
                
                for a in S_elems:
                    # Case 1: 2w = z + a
                    s1 = z + a
                    if s1 % 2 == 0:
                        w1 = s1 // 2
                        if 1 <= w1 <= N and w1 != z:
                            blocked_candidates.add(w1)
                            
                    # Case 2: w + a' = 2z (where a' is the other element in S, or z itself? 
                    # If we add z, the new sums include 2z. 
                    # If w+a' = 2z, and a' in S, then sum 2z is created by (w, a'). 
                    # But 2z is also created by (z, z). 
                    # So collision.
                    # Note: a' can be any element in S.
                    w2 = 2*z - a
                    if 1 <= w2 <= N and w2 != z:
                        blocked_candidates.add(w2)
                        
                    # Case 3: w + a' = z + a (where a' in S, a in S)
                    # Sum z+a is new (from z,a). Sum w+a' is new (from w,a').
                    # Wait, if w is added, w+a' is a new sum.
                    # If w+a' = z+a, then we have two new sums with same value?
                    # No, z is already added in the simulation.
                    # So z+a is a sum in the NEW set.
                    # If we add w, w+a' is a sum in the NEW set (if w were added).
                    # But we are checking if w is blocked by the set S U {z}.
                    # So we assume z is in S.
                    # The sums in S U {z} are Sigma U {2z} U {z+a | a in S}.
                    # w is blocked if adding w creates a collision with sums in S U {z}.
                    # Collision types:
                    # - 2w in Sums(S U {z})
                    # - w+a'' in Sums(S U {z}) for some a'' in S U {z}
                    
                    # My previous derivation:
                    # 1. 2w = 2z (w=z, excluded)
                    # 2. 2w = z+a => w=(z+a)/2
                    # 3. 2w = z+a' (same as 2)
                    
                    # 4. w + a'' = 2z (a'' in S U {z})
                    #    If a''=z, w=2z-z=z (excluded)
                    #    If a'' in S, w = 2z - a''
                    
                    # 5. w + a'' = z + a (a'' in S U {z}, a in S)
                    #    If a''=z, w+z = z+a => w=a (a in S, so w not in Avail)
                    #    If a'' in S, w+a'' = z+a => w = z + a - a''
                    
                    # So the candidates are indeed:
                    # w = (z+a)/2
                    # w = 2z - a
                    # w = z + a - a''
                    
                    for a_prime in S_elems:
                        w3 = z + a - a_prime
                        if 1 <= w3 <= N and w3 != z:
                            blocked_candidates.add(w3)
                
                # Count how many are in available_mask
                count = 0
                for w in blocked_candidates:
                    if (available_mask >> w) & 1:
                        count += 1
                
                score = count + random.random() * 0.1
                if score > max_score:
                    max_score = score
                    best_z = z
            
            if best_z == -1:
                break
            
            # Commit best_z
            S_mask |= (1 << best_z)
            S_list.append(best_z)
            
            # Update sums_mask
            sums_mask |= (1 << (2*best_z))
            for a in S_elems:
                sums_mask |= (1 << (best_z + a))
            
            # Update available_mask
            available_mask &= ~(1 << best_z)
            
            # Remove blocked elements from available_mask
            # We can just re-scan or use the candidate logic?
            # Re-scanning is O(|Avail|). 
            # Since we already computed candidates for best_z, we could use them?
            # But the set of blocked elements for best_z is exactly blocked_candidates intersected with Avail.
            # So we can just remove them.
            for w in blocked_candidates:
                if (available_mask >> w) & 1:
                    available_mask &= ~(1 << w)
            
            # Also, we must ensure best_z itself is removed (done above)
            # And we must check if any *other* elements become blocked that were not in blocked_candidates?
            # No, blocked_candidates contains ALL w that are blocked by adding best_z.
            # Proof: The conditions for w being blocked are derived exhaustively.
            
            # However, we must be careful: 
            # The set S_elems used for calculation was the OLD S.
            # After adding best_z, the new S is S U {best_z}.
            # The blocked set calculation was for "w blocked by S U {best_z}".
            # This is correct.
            
            # We need to update S_elems for the next iteration? 
            # S_elems is local to the loop, so it will be recalculated.
            
        if len(S_list) < best_size:
            best_size = len(S_list)
            best_set = S_list[:]
            
    return best_set
