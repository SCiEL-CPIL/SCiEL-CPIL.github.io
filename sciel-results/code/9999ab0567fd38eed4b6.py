import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 80
    
    n = 128
    target_lo = 128
    target_hi = 256
    n_targets = target_hi - target_lo + 1  # 129
    
    rng = np.random.default_rng(42)
    
    def mults_from_elems(elems):
        mults = np.zeros(n_targets, dtype=np.int32)
        k = len(elems)
        for i in range(k):
            for j in range(i, k):
                s = elems[i] + elems[j]
                if target_lo <= s <= target_hi:
                    mults[s - target_lo] += 1
        return mults
    
    def find_initial(trials=500):
        best = None
        best_peak = float('inf')
        for _ in range(trials):
            if time.time() > deadline:
                break
            k = int(rng.integers(16, 25))
            elems = np.sort(rng.choice(n + 1, size=k, replace=False))
            m = mults_from_elems(elems)
            if m.min() == 0:
                continue
            p = m.max()
            if p < best_peak:
                best_peak = p
                best = elems
        return best, best_peak
    
    elems, peak = find_initial()
    if elems is None:
        return list(range(64, 129))
    
    mults = mults_from_elems(elems)
    current_peak = mults.max()
    
    T = 3.0
    T_min = 0.005
    cooling = 0.9999
    
    while time.time() < deadline and T > T_min:
        k = len(elems)
        
        if k < 3:
            x = int(rng.integers(0, n + 1))
            if x not in elems:
                pos = np.searchsorted(elems, x)
                new_elems = np.insert(elems, pos, x)
                nm = mults_from_elems(new_elems)
                if nm.min() > 0:
                    npk = nm.max()
                    d = npk - current_peak
                    if d <= 0 or rng.random() < np.exp(-d / T):
                        elems = new_elems
                        mults = nm
                        current_peak = npk
        else:
            rem_i = int(rng.integers(0, k))
            x = int(elems[rem_i])
            x2 = int(rng.integers(0, n + 1))
            if x2 == x:
                T *= cooling
                continue
            mask = elems != x
            base = elems[mask]
            pos = np.searchsorted(base, x2)
            new_elems = np.insert(base, pos, x2)
            nm = mults_from_elems(new_elems)
            if nm.min() == 0:
                T *= cooling
                continue
            npk = nm.max()
            d = npk - current_peak
            if d <= 0 or rng.random() < np.exp(-d / T):
                elems = new_elems
                mults = nm
                current_peak = npk
        
        T *= cooling
    
    final_mults = mults_from_elems(elems)
    if final_mults.min() == 0:
        return list(range(64, 129))
    return list(elems)
