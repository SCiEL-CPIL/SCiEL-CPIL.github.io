import time
import random
import numpy as np

def construct():
    n = 256
    start_time = time.time()
    time_limit = 55.0  # Safety margin for 60s limit
    
    # Precompute all 3-APs and 4-APs as lists of indices (0-indexed)
    # 3-AP: (a, a+d, a+2d)
    ap3s = []
    for d in range(1, n // 2 + 1):
        for a in range(0, n - 2 * d):
            ap3s.append((a, a + d, a + 2 * d))
    
    # 4-AP: (a, a+d, a+2d, a+3d)
    ap4s = []
    for d in range(1, n // 3 + 1):
        for a in range(0, n - 3 * d):
            ap4s.append((a, a + d, a + 2 * d, a + 3 * d))
    
    # Convert to numpy arrays for vectorized operations
    # ap3_array: shape (M3, 3)
    # ap4_array: shape (M4, 4)
    ap3_array = np.array(ap3s, dtype=np.int32)
    ap4_array = np.array(ap4s, dtype=np.int32)
    
    M3 = len(ap3_array)
    M4 = len(ap4_array)
    
    # For each element v, find which 3-APs and 4-APs it participates in
    # This allows us to quickly check validity and compute deltas
    # ap3_membership[v] = list of indices in ap3_array where v is present
    # ap4_membership[v] = list of indices in ap4_array where v is present
    
    ap3_membership = [[] for _ in range(n)]
    ap4_membership = [[] for _ in range(n)]
    
    for idx, ap in enumerate(ap3_array):
        for v in ap:
            ap3_membership[v].append(idx)
            
    for idx, ap in enumerate(ap4_array):
        for v in ap:
            ap4_membership[v].append(idx)
    
    # Convert membership lists to numpy arrays for faster access
    ap3_mem_arrays = [np.array(lst, dtype=np.int32) if lst else np.array([], dtype=np.int32) for lst in ap3_membership]
    ap4_mem_arrays = [np.array(lst, dtype=np.int32) if lst else np.array([], dtype=np.int32) for lst in ap4_membership]
    
    def count_3aps(in_S):
        """
        Count number of 3-APs fully contained in S.
        in_S: boolean numpy array of length n
        """
        if M3 == 0:
            return 0
        # Check if all 3 elements of each 3-AP are in S
        # ap3_array is (M3, 3). in_S[ap3_array] gives (M3, 3) boolean array
        # all(axis=1) gives (M3,) boolean array
        active = np.all(in_S[ap3_array], axis=1)
        return np.sum(active)
    
    def is_valid_add(in_S, v):
        """
        Check if adding v to S (represented by in_S, where v is False) 
        results in a 4-AP-free set.
        We need to ensure no 4-AP containing v has its other 3 elements in S.
        """
        mem = ap4_mem_arrays[v]
        if len(mem) == 0:
            return True
        
        # Get the 4-APs containing v
        # ap4_array[mem] is (K, 4)
        # We need to check if the other 3 elements are in S.
        # Since v is not in S yet, in_S[v] is False.
        # We can just check if all 4 elements are in S? No, v is not in S.
        # So we check if the other 3 are in S.
        
        # Let's extract the 4-APs
        aps = ap4_array[mem] # (K, 4)
        
        # For each AP, find the index of v and mask it out?
        # Or simpler: check if (in_S[aps] & ~v_mask).all(axis=1)
        # Actually, since v is not in S, in_S[v] is 0.
        # We want to know if the other 3 are 1.
        # If any AP has 3 ones and the v-position is 0, then adding v completes a 4-AP.
        
        # Let's create a boolean matrix where True means "element is in S"
        # in_S[aps] gives the status of the 4 elements for each AP.
        # We want to know if the sum of in_S[aps] is 3 (since v is 0, and we need the other 3 to be 1).
        # Wait, if v is 0, and the other 3 are 1, the sum is 3.
        # If v is 1 (which it isn't), sum would be 4.
        
        statuses = in_S[aps] # (K, 4)
        # Count how many are in S for each AP
        counts = np.sum(statuses, axis=1)
        
        # If any count is 3, it means the other 3 are in S, so adding v creates a 4-AP.
        return np.all(counts < 3)
        
    def calc_delta(in_S, u, v):
        """
        Calculate the change in 3-AP count if we swap u (in S) with v (not in S).
        u is currently in S, v is not.
        New S = S \ {u} U {v}
        
        Delta = (3-APs in New S) - (3-APs in Old S)
        = (3-APs in S that don't involve u or v) 
          + (3-APs involving v but not u, where other elements in S\{u})
          - (3-APs involving u but not v, where other elements in S\{v})
        
        Note: 3-APs involving both u and v are removed from S (since u is removed) 
        and not added back (since v doesn't form them with u). 
        Actually, if a 3-AP has both u and v, it is in Old S. In New S, u is gone, so it's gone.
        So we subtract all 3-APs containing u that are active in Old S.
        And add all 3-APs containing v that are active in New S (which means other elements in S\{u}).
        
        Let's refine:
        Lost: 3-APs containing u, where the other 2 elements are in S.
        Gained: 3-APs containing v, where the other 2 elements are in S \ {u}.
        
        Since v is not in S, in_S[v] is False.
        """
        
        # Lost: 3-APs containing u
        mem_u = ap3_mem_arrays[u]
        if len(mem_u) > 0:
            aps_u = ap3_array[mem_u] # (Ku, 3)
            # Check if other 2 elements are in S
            # in_S[aps_u] is (Ku, 3). u is in S, so one column is 1.
            # We want to know if the other two are 1.
            # Sum of in_S[aps_u] should be 3 (u=1, others=1).
            counts_u = np.sum(in_S[aps_u], axis=1)
            lost = np.sum(counts_u == 3)
        else:
            lost = 0
            
        # Gained: 3-APs containing v
        mem_v = ap3_mem_arrays[v]
        if len(mem_v) > 0:
            aps_v = ap3_array[mem_v] # (Kv, 3)
            # We need the other 2 elements to be in S \ {u}.
            # in_S[aps_v] is (Kv, 3). v is not in S, so one column is 0.
            # We want the other two to be 1.
            # However, if one of the other two is u, and u is in S, 
            # but we are removing u, so that 3-AP is NOT gained.
            # So we need to check:
            # 1. The other two elements are in S.
            # 2. The other two elements are NOT u.
            
            # Let's get the indices of the other two elements for each AP containing v.
            # For each AP (x, y, z) in aps_v, one of them is v.
            # We need to check the other two.
            
            # Vectorized approach:
            # Create a mask for "is in S" and "is not u"
            # in_S is boolean. 
            # We want to count APs where:
            # - v is in the AP (guaranteed by mem_v)
            # - The other two elements are in S
            # - The other two elements are not u
            
            # Let's create a modified in_S where u is False.
            in_S_no_u = in_S.copy()
            in_S_no_u[u] = False
            
            # Now check if the other two elements are in in_S_no_u.
            # Since v is not in in_S_no_u (it wasn't in in_S), 
            # we can just check if the sum of in_S_no_u[aps_v] is 2.
            # Wait, if the AP is (v, a, b), in_S_no_u[v]=0, in_S_no_u[a]=?, in_S_no_u[b]=?.
            # If a,b in S and a,b != u, then sum is 2.
            # If a=u, in_S_no_u[a]=0, sum is 1.
            # If a,b in S but one is u, sum is 1.
            
            counts_v = np.sum(in_S_no_u[aps_v], axis=1)
            gained = np.sum(counts_v == 2)
        else:
            gained = 0
            
        return gained - lost
        
    best_count = 0
    best_in_S = np.zeros(n, dtype=bool)
    
    # Multi-Start Local Search
    num_restarts = 10
    
    for restart in range(num_restarts):
        if time.time() - start_time > time_limit:
            break
            
        # 1. Initial Greedy Construction
        in_S = np.zeros(n, dtype=bool)
        S_set = set()
        
        candidates = list(range(n))
        if restart == 0:
            pass
        else:
            random.shuffle(candidates)
            
        for x in candidates:
            if is_valid_add(in_S, x):
                in_S[x] = True
                S_set.add(x)
                
        current_count = count_3aps(in_S)
        
        if current_count > best_count:
            best_count = current_count
            best_in_S = in_S.copy()
            
        # 2. Local Search (Hill Climbing)
        improved = True
        while improved and (time.time() - start_time < time_limit):
            improved = False
            S_list = [i for i in range(n) if in_S[i]]
            N_list = [i for i in range(n) if not in_S[i]]
            
            # To speed up, we can iterate through swaps.
            # For each u in S, check all v in N.
            # This is |S| * |N| checks.
            # Each check involves is_valid_add (vectorized) and calc_delta (vectorized).
            # is_valid_add is fast if mem is small.
            # calc_delta is fast if mem is small.
            
            # Optimization: Pre-compute validity for all v in N?
            # Validity of v depends on S. S changes after each swap.
            # So we must re-check.
            
            # Let's try to be smarter.
            # We can iterate u, and for each u, find the best v.
            
            for u in S_list:
                if time.time() - start_time > time_limit:
                    break
                    
                best_v = -1
                best_delta = 0
                
                # Check all v in N_list
                # We can vectorize the validity check for all v?
                # is_valid_add checks if any 4-AP containing v has other 3 in S\{u}.
                # This is hard to vectorize across v because the 4-APs differ.
                
                # Let's just loop v, but use the fast numpy functions.
                for v in N_list:
                    # Quick check: if v is not valid, skip
                    # We need to check validity in S \ {u}
                    # Let's create a temporary in_S_no_u
                    # Actually, is_valid_add assumes v is not in S.
                    # But it checks against the current in_S.
                    # We need to check against S \ {u}.
                    
                    # Modify in_S temporarily? No, copy is expensive.
                    # Let's write a specialized is_valid_add_in_S_minus_u(in_S, u, v)
                    
                    mem_v_4 = ap4_mem_arrays[v]
                    if len(mem_v_4) > 0:
                        aps_v = ap4_array[mem_v_4]
                        # We need other 3 elements in S \ {u}
                        # in_S[aps_v] gives status.
                        # Set u to 0 in the check?
                        # We can do:
                        # counts = sum(in_S[aps_v], axis=1)
                        # But we must subtract 1 if u is in the AP and in_S[u] is 1.
                        # Actually, u is in S.
                        # If u is in the AP, and we remove u, the count of "other elements in S" decreases by 1.
                        
                        # Let's just use the logic from calc_delta but for validity.
                        # Valid if no 4-AP has 3 other elements in S\{u}.
                        
                        in_S_no_u = in_S.copy()
                        in_S_no_u[u] = False
                        
                        counts = np.sum(in_S_no_u[aps_v], axis=1)
                        # v is not in in_S_no_u.
                        # If any count is 3, invalid.
                        if np.any(counts == 3):
                            continue
                            
                    # Calculate delta
                    delta = calc_delta(in_S, u, v)
                    
                    if delta > best_delta:
                        best_delta = delta
                        best_v = v
                        
                if best_v != -1:
                    # Perform swap
                    in_S[u] = False
                    in_S[best_v] = True
                    S_set.remove(u)
                    S_set.add(best_v)
                    current_count += best_delta
                    improved = True
                    
                    if current_count > best_count:
                        best_count = current_count
                        best_in_S = in_S.copy()
                        
                    # Break out of u loop to restart with new S
                    break
                    
    # Convert best_in_S to list
    if not np.any(best_in_S):
        return [1]
        
    result = [i + 1 for i in range(n) if best_in_S[i]]
    return result

if __name__ == "__main__":
    res = construct()
    print(len(res), res[:10])
