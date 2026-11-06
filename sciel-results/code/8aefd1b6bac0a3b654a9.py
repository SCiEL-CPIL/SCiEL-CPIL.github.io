import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 80.0
    
    n = 128
    
    # Precompute all 4-term APs as index arrays
    # (i, i+d, i+2d, i+3d) for d>=1, i+3d < n
    ap_list = []
    for d in range(1, n // 3 + 1):
        max_i = n - 3 * d
        for i in range(max_i):
            ap_list.append((i, i + d, i + 2 * d, i + 3 * d))
    
    ap_arr = np.array(ap_list)  # shape (num_aps, 4)
    num_aps = len(ap_arr)
    
    def check_valid(colors):
        """Vectorized check: every 4-term AP has >= 3 distinct colors."""
        # For each AP, get the 4 colors
        ap_colors = colors[ap_arr]  # shape (num_aps, 4)
        # Count distinct colors: for each row, check if all same (1 color) or exactly 2 colors
        # All same: all 4 equal
        all_same = np.all(ap_colors == ap_colors[:, 0:1], axis=1)
        # Exactly 2 colors: not all same, but only 2 distinct
        # A row has <=2 distinct colors iff for some pair, the other two are in {those two}
        # Simpler: a row has >=3 distinct iff it's not the case that all elements are in some 2-element set
        # Check: for each row, are there at least 3 distinct?
        # A row has <3 distinct colors iff all elements belong to a set of size <=2
        # Equivalently: the row is NOT "all distinct in 3+"
        # Let's check: sort each row, then check if c0==c3 or (c0==c1 and c2==c3) or (c0==c1==c2 or c1==c2==c3) etc.
        # Simpler: a row has <3 distinct iff (c0,c1,c2,c3) has at most 2 unique values
        # Check: (c0==c1 and c0==c2 and c0==c3) OR (c0==c1 and c2==c3) OR (c0==c2 and c1==c3) OR (c0==c3 and c1==c2) OR (c0==c2 and c0==c3) etc.
        # Actually: <3 distinct means the set of values has size 1 or 2.
        # Size 1: all equal
        # Size 2: exactly two values, so the multiset is {a,a,a,b}, {a,a,b,b}, or {a,a,b,a} etc.
        # A row has exactly 2 distinct values iff not all same AND (after sorting, c0==c1 and c2==c3) OR (c0==c1==c2 and c3!=c0) OR (c0!=c1 and c1==c2==c3)
        # Let me just compute: for each row, the number of distinct values.
        # A row has <3 distinct iff:
        #   all 4 same, OR
        #   3 same + 1 different, OR  
        #   2+2 split
        # Check: (c0==c1==c2==c3) or (c0==c1==c2 and c3!=c0) or (c0==c1==c3 and c2!=c0) or (c0==c1==c2... 
        # Simpler approach: a row has >=3 distinct iff it is NOT the case that all elements are in {x,y} for some x,y.
        # Equivalent: the row has >=3 distinct iff (c0 != c1 or c0 != c2 or c0 != c3) AND (c1 != c2 or c1 != c3 or c2 != c3) AND ... this is getting complex.
        
        # Let me just do it directly:
        # A row has <3 distinct colors iff:
        #   (c0==c1 and c0==c2 and c0==c3)  # all same
        #   or (c0==c1 and c0==c2 and c2!=c3)  # 3+1
        #   or (c0==c1 and c0==c3 and c2!=c0)  # 3+1
        #   or (c0==c2 and c0==c3 and c1!=c0)  # 3+1
        #   or (c1==c2 and c1==c3 and c0!=c1)  # 3+1
        #   or (c0==c1 and c2==c3 and c0!=c2)  # 2+2
        #   or (c0==c2 and c1==c3 and c0!=c1)  # 2+2
        #   or (c0==c3 and c1==c2 and c0!=c1)  # 2+2
        
        c0, c1, c2, c3 = ap_colors[:,0], ap_colors[:,1], ap_colors[:,2], ap_colors[:,3]
        
        all_same = (c0 == c1) & (c1 == c2) & (c2 == c3)
        three_one_1 = (c0 == c1) & (c1 == c2) & (c2 != c3)
        three_one_2 = (c0 == c1) & (c1 == c3) & (c2 != c0)
        three_one_3 = (c0 == c2) & (c2 == c3) & (c1 != c0)
        three_one_4 = (c1 == c2) & (c2 == c3) & (c0 != c1)
        two_two_1 = (c0 == c1) & (c2 == c3) & (c0 != c2)
        two_two_2 = (c0 == c2) & (c1 == c3) & (c0 != c1)
        two_two_3 = (c0 == c3) & (c1 == c2) & (c0 != c1)
        
        bad = (all_same | three_one_1 | three_one_2 | three_one_3 | three_one_4 | 
               two_two_1 | two_two_2 | two_two_3)
        return not np.any(bad)
    
    def count_violations(colors):
        ap_colors = colors[ap_arr]
        c0, c1, c2, c3 = ap_colors[:,0], ap_colors[:,1], ap_colors[:,2], ap_colors[:,3]
        all_same = (c0 == c1) & (c1 == c2) & (c2 == c3)
        three_one_1 = (c0 == c1) & (c1 == c2) & (c2 != c3)
        three_one_2 = (c0 == c1) & (c1 == c3) & (c2 != c0)
        three_one_3 = (c0 == c2) & (c2 == c3) & (c1 != c0)
        three_one_4 = (c1 == c2) & (c2 == c3) & (c0 != c1)
        two_two_1 = (c0 == c1) & (c2 == c3) & (c0 != c2)
        two_two_2 = (c0 == c2) & (c1 == c3) & (c0 != c1)
        two_two_3 = (c0 == c3) & (c1 == c2) & (c0 != c1)
        bad = (all_same | three_one_1 | three_one_2 | three_one_3 | three_one_4 | 
               two_two_1 | two_two_2 | two_two_3)
        return np.sum(bad)
    
    best_colors = None
    best_num_colors = n + 1
    
    # Try different numbers of colors
    for num_c in range(3, min(8, n + 1)):
        if time.time() > deadline - 10:
            break
        
        for attempt in range(200):
            if time.time() > deadline - 10:
                break
            
            # Random init
            colors = np.random.randint(0, num_c, n).astype(np.int32)
            
            # Local search
            improved = True
            iters = 0
            while improved and iters < 500:
                improved = False
                iters += 1
                
                if time.time() > deadline - 10:
                    break
                
                v = count_violations(colors)
                if v == 0:
                    break
                
                # Find a violating position
                # Get all violating APs
                ap_colors = colors[ap_arr]
                c0, c1, c2, c3 = ap_colors[:,0], ap_colors[:,1], ap_colors[:,2], ap_colors[:,3]
                all_same = (c0 == c1) & (c1 == c2) & (c2 == c3)
                three_one_1 = (c0 == c1) & (c1 == c2) & (c2 != c3)
                three_one_2 = (c0 == c1) & (c1 == c3) & (c2 != c0)
                three_one_3 = (c0 == c2) & (c2 == c3) & (c1 != c0)
                three_one_4 = (c1 == c2) & (c2 == c3) & (c0 != c1)
                two_two_1 = (c0 == c1) & (c2 == c3) & (c0 != c2)
                two_two_2 = (c0 == c2) & (c1 == c3) & (c0 != c1)
                two_two_3 = (c0 == c3) & (c1 == c2) & (c0 != c1)
                bad = (all_same | three_one_1 | three_one_2 | three_one_3 | three_one_4 | 
                       two_two_1 | two_two_2 | two_two_3)
                
                if not np.any(bad):
                    break
                
                # Pick random violating AP
                bad_indices = np.where(bad)[0]
                ap_idx = bad_indices[random.randint(0, len(bad_indices) - 1)]
                i, j, k, l = ap_arr[ap_idx]
                
                # Try changing each position
                positions = [i, j, k, l]
                random.shuffle(positions)
                
                best_change = None
                best_v = v
                
                for pos in positions:
                    old = colors[pos]
                    for new_c in range(num_c):
                        if new_c == old:
                            continue
                        colors[pos] = new_c
                        nv = count_violations(colors)
                        if nv < best_v:
                            best_v = nv
                            best_change = (pos, new_c)
                    colors[pos] = old
                
                if best_change:
                    pos, new_c = best_change
                    colors[pos] = new_c
                    improved = True
            
            if check_valid(colors):
                num_used = len(set(colors.tolist()))
                if num_used < best_num_colors:
                    best_num_colors = num_used
                    best_colors = colors.copy()
        
        if best_colors is not None and best_num_colors <= num_c:
            # Found a valid coloring, try to reduce
            break
    
    if best_colors is not None:
        result = [int(x) for x in best_colors]
        return result
    else:
        return list(range(n))
