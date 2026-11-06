import time
import random
import math

# Global precomputation
_N = 128
_AP_LIST = []
_MAX_D = (_N - 1) // 3
for _d in range(1, _MAX_D + 1):
    _max_a = _N - 3 * _d
    for _a in range(1, _max_a + 1):
        _AP_LIST.append((_a, _a + _d, _a + 2 * _d, _a + 3 * _d))

# Map position to AP indices
_POS_TO_APS = [[] for _ in range(_N)]
for _idx, _ap in enumerate(_AP_LIST):
    for _p in _ap:
        _POS_TO_APS[_p - 1].append(_idx)

def _count_violations(colors):
    v = 0
    for _ap in _AP_LIST:
        c0 = colors[_ap[0] - 1]
        c1 = colors[_ap[1] - 1]
        c2 = colors[_ap[2] - 1]
        c3 = colors[_ap[3] - 1]
        if len({c0, c1, c2, c3}) < 3:
            v += 1
    return v

def _local_search(k, time_limit):
    start_t = time.time()
    current_colors = [random.randint(0, k - 1) for _ in range(_N)]
    current_v = _count_violations(current_colors)
    
    if current_v == 0:
        return current_colors

    T = 1.0
    cooling = 0.999
    min_T = 0.001

    while time.time() - start_t < time_limit:
        if current_v == 0:
            return current_colors

        pos = random.randint(0, _N - 1)
        old_c = current_colors[pos]
        new_c = random.randint(0, k - 1)
        if new_c == old_c:
            continue

        delta_v = 0
        for ap_idx in _POS_TO_APS[pos]:
            ap = _AP_LIST[ap_idx]
            old_set = set()
            new_set = set()
            for p in ap:
                c_o = current_colors[p - 1]
                old_set.add(c_o)
                c_n = new_c if (p - 1 == pos) else c_o
                new_set.add(c_n)
            old_bad = 1 if len(old_set) < 3 else 0
            new_bad = 1 if len(new_set) < 3 else 0
            delta_v += new_bad - old_bad

        if delta_v < 0:
            current_colors[pos] = new_c
            current_v += delta_v
        else:
            if T > min_T:
                if random.random() < math.exp(-delta_v / T):
                    current_colors[pos] = new_c
                    current_v += delta_v
        T *= cooling

    return None

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 55.0 # 5s buffer
    
    for k in range(3, 13):
        if time.time() > deadline:
            break
        
        # Allocate time for this k
        # Smaller k is harder, give more time?
        # Or just fixed time per k.
        time_slice = 5.0
        if k == 3:
            time_slice = 10.0
        elif k == 4:
            time_slice = 10.0
            
        # Try to find a solution
        sol = _local_search(k, time_slice)
        
        if sol is not None:
            return sol
            
    # Fallback: unique colors (128 colors)
    return list(range(_N))
# EVOLVE-BLOCK-END
