import random
import time

def construct():
    N = 128
    # Precompute all 4-term APs
    # AP is (a, a+d, a+2d, a+3d) with 1 <= a and a+3d <= N
    aps = []
    # Map index to list of AP indices it belongs to
    index_to_aps = [[] for _ in range(N)]
    
    for a in range(1, N + 1):
        for d in range(1, N + 1):
            b = a + d
            c = a + 2 * d
            e = a + 3 * d
            if e > N:
                break
            ap = (a - 1, b - 1, c - 1, e - 1) # 0-indexed
            ap_idx = len(aps)
            aps.append(ap)
            for idx in ap:
                index_to_aps[idx].append(ap_idx)
                
    # Helper to check if an AP is bad (uses <= 2 distinct colors)
    def is_bad_ap(ap, colors):
        c1 = colors[ap[0]]
        c2 = colors[ap[1]]
        c3 = colors[ap[2]]
        c4 = colors[ap[3]]
        # Check distinct count
        # Fast check: if any two are equal, check if the other two match one of them or each other
        # Simpler: len(set(...)) but set creation is slow.
        # Manual check for size < 3
        if c1 == c2:
            if c3 == c4:
                # Pairs (c1,c1) and (c3,c3). Bad if c1==c3 (monochromatic) or c1!=c3 (2 colors)
                # Actually if c1==c2 and c3==c4, distinct colors are {c1, c3}. Size 1 or 2. Always bad.
                return True
            else:
                # c1,c1,c3,c4 with c3!=c4.
                # If c1==c3, colors are {c1, c4}. Size 2. Bad.
                # If c1==c4, colors are {c1, c3}. Size 2. Bad.
                # If c1!=c3 and c1!=c4, colors are {c1, c3, c4}. Size 3. Good.
                if c1 == c3 or c1 == c4:
                    return True
                else:
                    return False
        elif c1 == c3:
            # c1,c2,c1,c4. c1!=c2.
            # If c2==c4, colors {c1, c2}. Size 2. Bad.
            # If c2!=c4, colors {c1, c2, c4}. Size 3. Good.
            if c2 == c4:
                return True
            else:
                return False
        elif c1 == c4:
            # c1,c2,c3,c1. c1!=c2, c1!=c3.
            # If c2==c3, colors {c1, c2}. Size 2. Bad.
            # If c2!=c3, colors {c1, c2, c3}. Size 3. Good.
            if c2 == c3:
                return True
            else:
                return False
        else:
            # c1,c2,c3,c4 all distinct so far? No, c1!=c2,c3,c4.
            # Check c2,c3,c4.
            if c2 == c3:
                # c1,c2,c2,c4. c1!=c2.
                # If c2==c4, colors {c1, c2}. Bad.
                # If c2!=c4, colors {c1, c2, c4}. Good.
                if c2 == c4:
                    return True
                else:
                    return False
            elif c2 == c4:
                # c1,c2,c3,c2. c1!=c2, c1!=c3 (implied? No, c1!=c4=c2, c1!=c2. c1 vs c3 unknown).
                # Wait, c1!=c2, c1!=c4. c2==c4.
                # Colors: c1, c2, c3, c2.
                # If c1==c3, colors {c1, c2}. Bad.
                # If c1!=c3, colors {c1, c2, c3}. Good.
                if c1 == c3:
                    return True
                else:
                    return False
            elif c3 == c4:
                # c1,c2,c3,c3. c1!=c2, c1!=c3.
                # If c2==c3, colors {c1, c2}. Bad.
                # If c2!=c3, colors {c1, c2, c3}. Good.
                if c2 == c3:
                    return True
                else:
                    return False
            else:
                # All 4 distinct. Good.
                return False

    def calculate_initial_penalty(colors):
        total = 0
        ap_bad = [False] * len(aps)
        for i, ap in enumerate(aps):
            if is_bad_ap(ap, colors):
                total += 1
                ap_bad[i] = True
        return total, ap_bad

    def get_ap_penalty_for_index(idx, colors, ap_bad):
        # Returns number of bad APs involving idx
        count = 0
        for ap_idx in index_to_aps[idx]:
            if ap_bad[ap_idx]:
                count += 1
        return count

    def try_find_solution(K, time_limit=5.0):
        start_time = time.time()
        # Try multiple restarts
        max_restarts = 20
        for _ in range(max_restarts):
            if time.time() - start_time > time_limit:
                break
            
            colors = [random.randint(0, K - 1) for _ in range(N)]
            penalty, ap_bad = calculate_initial_penalty(colors)
            
            if penalty == 0:
                return colors
            
            # Local search
            # We maintain ap_bad and penalty
            # For each step, pick a random index, try all K colors, pick best
            # To speed up, we can prioritize indices with high local penalty
            # But random is often good enough for small N
            
            for step in range(10000):
                if time.time() - start_time > time_limit:
                    break
                
                # Pick a random index
                i = random.randint(0, N - 1)
                old_color = colors[i]
                
                best_color = old_color
                best_penalty = penalty
                
                for c in range(K):
                    if c == old_color:
                        continue
                    
                    # Calculate change in penalty
                    # Only APs involving i can change status
                    delta = 0
                    for ap_idx in index_to_aps[i]:
                        ap = aps[ap_idx]
                        was_bad = ap_bad[ap_idx]
                        
                        # Temporarily apply color
                        # We need to check if this AP is bad with new color at i
                        # Reconstruct colors for this AP? 
                        # Or just modify colors[i] temporarily?
                        # Modification is faster if we revert.
                        
                        colors[i] = c
                        is_now_bad = is_bad_ap(ap, colors)
                        
                        if was_bad and not is_now_bad:
                            delta -= 1
                        elif not was_bad and is_now_bad:
                            delta += 1
                        # Revert for next iteration
                        colors[i] = old_color
                        
                        # We need to know the new state of ap_bad for this ap to accumulate correctly?
                        # Actually, delta is sum of changes.
                        # But we need to update ap_bad array eventually.
                        # For now, just compute total delta.
                    
                    new_penalty = penalty + delta
                    if new_penalty < best_penalty:
                        best_penalty = new_penalty
                        best_color = c
                
                # Apply best color
                if best_color != old_color:
                    # Update ap_bad and penalty
                    for ap_idx in index_to_aps[i]:
                        ap = aps[ap_idx]
                        was_bad = ap_bad[ap_idx]
                        
                        colors[i] = best_color
                        is_now_bad = is_bad_ap(ap, colors)
                        
                        if was_bad and not is_now_bad:
                            penalty -= 1
                            ap_bad[ap_idx] = False
                        elif not was_bad and is_now_bad:
                            penalty += 1
                            ap_bad[ap_idx] = True
                        colors[i] = old_color # Revert, we will set it below
                    colors[i] = best_color
                    penalty = best_penalty
                    
                    if penalty == 0:
                        return colors

        return None

    # Try increasing number of colors
    for K in range(4, 10):
        sol = try_find_solution(K, time_limit=5.0)
        if sol:
            return sol
            
    # Fallback: Greedy from seed logic but optimized, or just return random if all fails
    # The seed logic is:
    # colors=[]
    # for x in range(1,129):
    #   for color in range(128):
    #     if (all(len({color}|{colors[x-1-j*d] for j in (1,2,3)})>=3 for d in range(1,(x-1)//3+1))
    #       and all(any(colors[x-1-j*d]!=color for j in (1,2)) for d in range(1,(x-1)//2+1))):
    #       colors.append(color);break
    # return colors
    
    # Let's just use a simple greedy that tries to reuse colors
    colors = []
    for x in range(1, 129):
        assigned = False
        for color in range(128):
            valid = True
            # Check all 4-term APs ending at x or containing x?
            # The seed checks APs where x is the LAST term?
            # "colors[x-1-j*d] for j in (1,2,3)" -> x-d, x-2d, x-3d.
            # So it only checks APs where x is the 4th term.
            # This is insufficient for global validity.
            # But the seed returns a valid result? "Every nonconstant four-term progression... must use AT LEAST three distinct colors."
            # If the seed is valid, it must have checked all APs.
            # The seed code:
            # all(len({color}|{colors[x-1-j*d] for j in (1,2,3)})>=3 for d in range(1,(x-1)//3+1))
            # This checks if adding `color` to the set of previous 3 terms in the AP keeps size >= 3.
            # This ensures the AP ending at x has >= 3 colors.
            # But what about APs where x is the 1st, 2nd, or 3rd term?
            # Those were checked when the later terms were assigned.
            # So greedy forward assignment works if we check all constraints involving x that are "closed" by x?
            # No, an AP $x, x+d, x+2d, x+3d$ is not closed until $x+3d$ is assigned.
            # So we cannot validate it yet.
            # The seed code seems to only check APs ending at x.
            # This implies the seed might NOT be valid for all APs?
            # Or maybe the condition "len({color} U prev) >= 3" is sufficient?
            # If we ensure every AP ending at i has >= 3 colors, does it imply all APs have >= 3?
            # No. An AP $a, a+d, a+2d, a+3d$ is checked when $a+3d$ is assigned.
            # So yes, every AP is checked exactly once, when its last element is assigned.
            # So the seed is valid.
            
            # My local search is better.
            # If local search fails, I can fall back to a slower greedy that uses more colors.
            
            # Let's implement a simple greedy with backtracking? No, too slow.
            # Just return a random valid coloring if found, else the seed's result.
            
            # For the fallback, let's just use the seed logic to ensure validity, even if score is 1.0.
            if not assigned:
                # Use seed logic for this step? No, seed logic is global.
                # Let's just try the next color in 0..127
                pass
        
        # This fallback is messy. 
        # Given N=128 and K=4/5, local search should almost certainly succeed.
        
    # If we reach here, return a dummy valid coloring using many colors
    # Re-run seed logic
    colors = []
    for x in range(1, 129):
        for color in range(128):
            if (all(len({color}|{colors[x-1-j*d] for j in (1,2,3)})>=3 for d in range(1,(x-1)//3+1))
                and all(any(colors[x-1-j*d]!=color for j in (1,2)) for d in range(1,(x-1)//2+1))):
                colors.append(color)
                break
    return colors
