import time
import random

# EVOLVE-BLOCK-START

def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    
    def try_construct(target_size, max_val=256, max_restarts=200):
        # We use a greedy approach with random restarts.
        # To speed up, we can limit the number of candidates we consider per restart
        # or just rely on the fact that valid sets are rare?
        
        for _ in range(max_restarts):
            if time.time() > deadline:
                return None
            
            candidates = list(range(1, max_val + 1))
            random.shuffle(candidates)
            
            S = []
            # sums[k] is a bitset
            sums = [1]  # sums[0] = 1
            
            for c in candidates:
                if len(S) == target_size:
                    break
                
                # Calculate new_sums
                # We only need to check up to len(S)+1
                new_sums = [0] * (len(S) + 2)
                
                # Optimization: Only compute what is needed?
                # new_sums[k] = sums[k] | (sums[k-1] << c)
                for k in range(len(S) + 1):
                    val = sums[k]
                    if k > 0:
                        val |= (sums[k-1] << c)
                    new_sums[k] = val
                
                # Check cross-cardinality collisions
                valid = True
                # We only need to check if new_sums intersects with others.
                # Since sums was valid, we only need to check new_sums[i] against new_sums[j]
                # where at least one is "new" (affected by c).
                # Actually, new_sums[i] = sums[i] | (sums[i-1] << c).
                # The "new" part is (sums[i-1] << c).
                # So we check:
                # 1. (sums[i-1] << c) & sums[j] == 0 for all j != i
                # 2. (sums[i-1] << c) & (sums[j-1] << c) == 0 for all j != i
                #    Note: (sums[i-1] << c) & (sums[j-1] << c) is (sums[i-1] & sums[j-1]) << c.
                #    Since sums was valid, sums[i-1] & sums[j-1] is 0 if i-1 != j-1.
                #    So condition 2 is automatically satisfied if i-1 != j-1.
                #    If i-1 == j-1, then i=j, which we don't check.
                #    So we ONLY need to check the new bits against the old bits.
                
                # Let's verify this logic.
                # new_sums[i] = Old_i | New_i
                # new_sums[j] = Old_j | New_j
                # We know Old_i & Old_j = 0 for i != j.
                # We need (Old_i | New_i) & (Old_j | New_j) = 0.
                # This expands to:
                # (Old_i & Old_j) | (Old_i & New_j) | (New_i & Old_j) | (New_i & New_j) = 0
                # Term 1: 0 (validity of S)
                # Term 4: (sums[i-1] << c) & (sums[j-1] << c) = (sums[i-1] & sums[j-1]) << c.
                #         If i != j, then i-1 != j-1, so sums[i-1] & sums[j-1] = 0.
                #         So Term 4 is 0.
                # We only need to check:
                # Old_i & New_j == 0  =>  sums[i] & (sums[j-1] << c) == 0
                # New_i & Old_j == 0  =>  (sums[i-1] << c) & sums[j] == 0
                
                # So for each i, j with i != j:
                # Check sums[i] & (sums[j-1] << c) == 0
                # Note: if j-1 > len(S), sums[j-1] is 0.
                # Also if i > len(S), sums[i] is 0.
                
                for i in range(len(S) + 1):
                    if sums[i] == 0: continue
                    for j in range(1, len(S) + 2):
                        if i == j: continue
                        if j - 1 > len(S): continue
                        if sums[j-1] == 0: continue
                        # Check collision
                        if sums[i] & (sums[j-1] << c):
                            valid = False
                            break
                    if not valid:
                        break
                
                if valid:
                    S.append(c)
                    sums = new_sums
            
            if len(S) == target_size:
                return S
        
        return None

    best_set = []
    
    # Try sizes from 12 down to 1
    # We can start higher, but 12 is a reasonable upper bound for 256.
    for size in range(12, 0, -1):
        if time.time() > deadline:
            break
        
        # Allocate more time for larger sizes?
        # Or just fixed restarts.
        res = try_construct(size, max_restarts=500)
        if res:
            best_set = res
            break
            
    if not best_set:
        best_set = [1]
        
    return best_set

# EVOLVE-BLOCK-END
