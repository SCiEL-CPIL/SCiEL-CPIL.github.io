import time
import random
import math

# Precomputation
N = 256

def get_ap_masks(term_count):
    masks = []
    max_d = (N - 1) // (term_count - 1)
    for d in range(1, max_d + 1):
        max_a = N - (term_count - 1) * d
        for a in range(1, max_a + 1):
            mask = 0
            for j in range(term_count):
                val = a + j * d
                mask |= (1 << (val - 1))
            masks.append(mask)
    return masks

M3 = get_ap_masks(3)
M4 = get_ap_masks(4)

# Precompute 3-APs and 4-APs containing each element x (0-indexed)
ap3_by_elem = [[] for _ in range(N)]
ap4_by_elem = [[] for _ in range(N)]

for i, m in enumerate(M3):
    for x in range(N):
        if (m >> x) & 1:
            ap3_by_elem[x].append(i)

for i, m in enumerate(M4):
    for x in range(N):
        if (m >> x) & 1:
            ap4_by_elem[x].append(i)

def construct():
    start_time = time.time()
    deadline = 45.0 # Safety margin
    
    # Initial Solution: Greedy
    best_mask = 0
    best_count = 0
    
    # Helper for Greedy
    def greedy_init():
        mask = 0
        count = 0
        order = list(range(N))
        random.shuffle(order)
        
        for x in order:
            bit = 1 << x
            if mask & bit:
                continue
            # Check validity
            valid = True
            for idx in ap4_by_elem[x]:
                m = M4[idx]
                if (m & ~bit & mask) == (m & ~bit):
                    valid = False
                    break
            if not valid:
                continue
            
            # Count gain
            delta = 0
            for idx in ap3_by_elem[x]:
                m = M3[idx]
                if (m & ~bit & mask) == (m & ~bit):
                    delta += 1
            
            mask |= bit
            count += delta
            
        return mask, count

    # Run 20 greedy inits
    for _ in range(20):
        m, c = greedy_init()
        if c > best_count:
            best_count = c
            best_mask = m

    # Simulated Annealing
    mask = best_mask
    count = best_count
    best_mask_sa = best_mask
    best_count_sa = best_count
    
    T = 5.0
    T_min = 0.01
    cooling = 0.9995
    bits = [1 << i for i in range(N)]
    
    while time.time() - start_time < deadline:
        x = random.randint(0, N - 1)
        bit = bits[x]
        
        if mask & bit:
            # Remove x
            delta = 0
            for idx in ap3_by_elem[x]:
                m = M3[idx]
                if (m & ~bit & mask) == (m & ~bit):
                    delta += 1
            
            # delta is number of 3-APs lost. Gain = -delta.
            if delta == 0:
                # Accept to wander
                mask &= ~bit
                count -= 0
            else:
                gain = -delta
                # Accept if gain > 0 (impossible here as delta>0) or with prob exp(gain/T)
                if random.random() < math.exp(gain / T):
                    mask &= ~bit
                    count += gain
                    if count > best_count_sa:
                        best_count_sa = count
                        best_mask_sa = mask
        else:
            # Add x
            valid = True
            for idx in ap4_by_elem[x]:
                m = M4[idx]
                if (m & ~bit & mask) == (m & ~bit):
                    valid = False
                    break
            
            if not valid:
                continue
                
            delta = 0
            for idx in ap3_by_elem[x]:
                m = M3[idx]
                if (m & ~bit & mask) == (m & ~bit):
                    delta += 1
            
            if delta == 0:
                mask |= bit
                count += 0
            else:
                if delta > 0 or random.random() < math.exp(delta / T):
                    mask |= bit
                    count += delta
                    if count > best_count_sa:
                        best_count_sa = count
                        best_mask_sa = mask
        
        T *= cooling
        if T < T_min:
            T = T_min
            # Reheat to allow larger jumps
            T = 1.0

    # Use the best found
    final_mask = best_mask_sa
    
    result = []
    for i in range(N):
        if final_mask & (1 << i):
            result.append(i + 1)
    return sorted(result)

if __name__ == "__main__":
    res = construct()
    print(f"Size: {len(res)}")
