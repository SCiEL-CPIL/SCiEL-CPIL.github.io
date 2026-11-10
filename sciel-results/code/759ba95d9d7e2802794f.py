import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 85  # safety margin before 90s kill

    N = 128
    T_START = 128
    T_END = 256
    NUM_TARGETS = T_END - T_START + 1  # 129 targets
    TARGETS = list(range(T_START, T_END + 1))

    # Precompute for each target t, the list of (a, b) pairs with a+b=t, a<=b, a,b in [0,N]
    pair_lists = {}
    for t in TARGETS:
        pairs = []
        for a in range(0, t // 2 + 1):
            b = t - a
            if 0 <= b <= N:
                pairs.append((a, b))
        pair_lists[t] = pairs

    def get_mults(A_set):
        """Return array of multiplicities for all targets."""
        A_list = sorted(A_set)
        mults = np.zeros(NUM_TARGETS, dtype=int)
        for i, t in enumerate(TARGETS):
            count = 0
            for a in A_list:
                if a > t // 2:
                    break
                b = t - a
                if b in A_set:
                    count += 1
            mults[i] = count
        return mults

    def get_mults_fast(A_bool):
        """A_bool is a numpy boolean array of length N+1. Return mults array."""
        mults = np.zeros(NUM_TARGETS, dtype=np.int32)
        for i, t in enumerate(TARGETS):
            count = 0
            for a, b in pair_lists[t]:
                if A_bool[a] and A_bool[b]:
                    count += 1
            mults[i] = count
        return mults

    def greedy_construct():
        """Build a valid set using greedy approach."""
        A = set()
        # Start with a reasonable base
        # We need every t in [128,256] to be representable
        # A simple valid set: all of [64, 128] works since a+b ranges [128, 256]
        # But we want smaller. Let's try a structured approach.
        
        # Greedy: for each target, if not covered, add elements
        mults = np.zeros(NUM_TARGETS, dtype=int)
        
        # Initial seed: try to use a mix
        # Add elements that cover many targets
        for t in TARGETS:
            if mults[t - T_START] > 0:
                continue
            # Find a pair (a,b) with a+b=t that maximizes coverage
            best_a, best_b = None, None
            best_gain = 0
            for a, b in pair_lists[t]:
                gain = 0
                if a not in A:
                    gain += 1
                if b not in A:
                    gain += 1
                # Count how many other targets this would help
                for t2 in TARGETS:
                    if mults[t2 - T_START] > 0:
                        continue
                    for a2, b2 in pair_lists[t2]:
                        if (a2 == a or b2 == a) and (a2 in A or b2 in A or a2 == a or b2 == a):
                            if a2 not in A or b2 not in A:
                                if a2 == a or b2 == a:
                                    if (a2 in A or a2 == a) and (b2 in A or b2 == b or b2 == a):
                                        gain += 1
                                    break
                        elif (a2 == b or b2 == b) and (a2 in A or b2 in A or a2 == b or b2 == b):
                            if a2 not in A or b2 not in A:
                                if (a2 in A or a2 == b) and (b2 in A or b2 == b or b2 == a):
                                    gain += 1
                                break
                if gain > best_gain:
                    best_gain = gain
                    best_a, best_b = a, b
            if best_a is not None:
                A.add(best_a)
                A.add(best_b)
                # Update mults
                for t2 in TARGETS:
                    for a2, b2 in pair_lists[t2]:
                        if a2 in A and b2 in A:
                            mults[t2 - T_START] += 1
                mults = get_mults(A)
        
        return A

    def ensure_valid(A_set):
        """Ensure A_set covers all targets. Add elements if needed."""
        A_set = set(A_set)
        while True:
            mults = get_mults(A_set)
            if np.min(mults) > 0:
                break
            # Find missing targets
            missing = [t for t, m in zip(TARGETS, mults) if m == 0]
            if not missing:
                break
            t = missing[0]
            # Find a pair that covers the most missing targets
            best_pair = None
            best_gain = 0
            for a, b in pair_lists[t]:
                gain = 1  # covers t
                for t2 in missing[1:]:
                    for a2, b2 in pair_lists[t2]:
                        if (a2 == a or b2 == a) and (b2 in A_set or b2 == b):
                            if a2 in A_set or a2 == a:
                                gain += 1
                                break
                        elif (a2 == b or b2 == b) and (a2 in A_set or a2 == a):
                            if b2 in A_set or b2 == b:
                                gain += 1
                                break
                if gain > best_gain:
                    best_gain = gain
                    best_pair = (a, b)
            if best_pair:
                A_set.add(best_pair[0])
                A_set.add(best_pair[1])
            else:
                # Fallback: just add t//2 and t - t//2
                a = t // 2
                b = t - a
                A_set.add(a)
                A_set.add(b)
        return A_set

    def simulated_annealing(A_set, max_iters=5000):
        """SA to minimize peak multiplicity."""
        A_set = set(A_set)
        A_bool = np.zeros(N + 1, dtype=bool)
        for a in A_set:
            A_bool[a] = True
        
        mults = get_mults_fast(A_bool)
        current_peak = np.max(mults)
        current_size = len(A_set)
        
        best_A = set(A_set)
        best_peak = current_peak
        
        T = 2.0  # initial temperature
        T_min = 0.01
        cooling = 0.995
        
        for it in range(max_iters):
            if time.time() > deadline - 3:
                break
            
            # Choose a random move type
            move_type = random.random()
            
            if move_type < 0.3 and len(A_set) > 10:
                # Removal
                a = random.choice(list(A_set))
                if not A_bool[a]:
                    continue
                # Compute delta mults
                old_mults = mults.copy()
                A_bool[a] = False
                new_mults = get_mults_fast(A_bool)
                
                if np.min(new_mults) == 0:
                    A_bool[a] = True
                    mults = old_mults
                    continue
                
                new_peak = np.max(new_mults)
                delta_peak = new_peak - current_peak
                
                if delta_peak <= 0 or random.random() < np.exp(-delta_peak / T):
                    A_set.remove(a)
                    mults = new_mults
                    current_peak = new_peak
                    current_size -= 1
                    if current_peak < best_peak:
                        best_peak = current_peak
                        best_A = set(A_set)
                else:
                    A_bool[a] = True
                    mults = old_mults
                    
            elif move_type < 0.7:
                # Swap: remove one, add one
                if len(A_set) == 0:
                    continue
                a_out = random.choice(list(A_set))
                a_in = random.randint(0, N)
                if a_in in A_set:
                    continue
                
                old_mults = mults.copy()
                A_bool[a_out] = False
                A_bool[a_in] = True
                new_mults = get_mults_fast(A_bool)
                
                if np.min(new_mults) == 0:
                    A_bool[a_out] = True
                    A_bool[a_in] = False
                    mults = old_mults
                    continue
                
                new_peak = np.max(new_mults)
                delta_peak = new_peak - current_peak
                
                if delta_peak <= 0 or random.random() < np.exp(-delta_peak / T):
                    A_set.remove(a_out)
                    A_set.add(a_in)
                    mults = new_mults
                    current_peak = new_peak
                    if current_peak < best_peak:
                        best_peak = current_peak
                        best_A = set(A_set)
                else:
                    A_bool[a_out] = True
                    A_bool[a_in] = False
                    mults = old_mults
                    
            elif move_type < 0.9:
                # Addition
                a_in = random.randint(0, N)
                if a_in in A_set:
                    continue
                
                old_mults = mults.copy()
                A_bool[a_in] = True
                new_mults = get_mults_fast(A_bool)
                
                new_peak = np.max(new_mults)
                delta_peak = new_peak - current_peak
                
                # Addition increases size, only accept if it reduces peak significantly
                if delta_peak < 0 or (delta_peak == 0 and random.random() < 0.1):
                    A_set.add(a_in)
                    mults = new_mults
                    current_peak = new_peak
                    current_size += 1
                    if current_peak < best_peak:
                        best_peak = current_peak
                        best_A = set(A_set)
                else:
                    A_bool[a_in] = False
                    mults = old_mults
                    
            else:
                # Double swap
                if len(A_set) < 2:
                    continue
                a1_out = random.choice(list(A_set))
                a2_out = random.choice([x for x in A_set if x != a1_out])
                a1_in = random.randint(0, N)
                a2_in = random.randint(0, N)
                if a1_in in A_set or a2_in in A_set or a1_in == a2_in:
                    continue
                
                old_mults = mults.copy()
                A_bool[a1_out] = False
                A_bool[a2_out] = False
                A_bool[a1_in] = True
                A_bool[a2_in] = True
                new_mults = get_mults_fast(A_bool)
                
                if np.min(new_mults) == 0:
                    A_bool[a1_out] = True
                    A_bool[a2_out] = True
                    A_bool[a1_in] = False
                    A_bool[a2_in] = False
                    mults = old_mults
                    continue
                
                new_peak = np.max(new_mults)
                delta_peak = new_peak - current_peak
                
                if delta_peak <= 0 or random.random() < np.exp(-delta_peak / T):
                    A_set.remove(a1_out)
                    A_set.remove(a2_out)
                    A_set.add(a1_in)
                    A_set.add(a2_in)
                    mults = new_mults
                    current_peak = new_peak
                    if current_peak < best_peak:
                        best_peak = current_peak
                        best_A = set(A_set)
                else:
                    A_bool[a1_out] = True
                    A_bool[a2_out] = True
                    A_bool[a1_in] = False
                    A_bool[a2_in] = False
                    mults = old_mults
            
            T *= cooling
            if T < T_min:
                T = T_min
        
        return best_A, best_peak

    # Generate multiple initial solutions
    best_A = None
    best_peak = float('inf')
    
    # Seed 1: Full range [64,128] - always valid
    A1 = set(range(64, 129))
    mults1 = get_mults(A1)
    if np.min(mults1) > 0:
        A1_sa, peak1 = simulated_annealing(A1, max_iters=2000)
        if peak1 < best_peak:
            best_peak = peak1
            best_A = A1_sa
    
    # Seed 2: Structured - every 2nd element
    A2 = set(range(0, N + 1, 2))
    A2 = ensure_valid(A2)
    mults2 = get_mults(A2)
    if np.min(mults2) > 0:
        A2_sa, peak2 = simulated_annealing(A2, max_iters=2000)
        if peak2 < best_peak:
            best_peak = peak2
            best_A = A2_sa
    
    # Seed 3: Random with coverage
    random.seed(42)
    for trial in range(3):
        k = random.randint(25, 45)
        A3 = set(random.sample(range(0, N + 1), k))
        A3 = ensure_valid(A3)
        mults3 = get_mults(A3)
        if np.min(mults3) > 0:
            A3_sa, peak3 = simulated_annealing(A3, max_iters=1500)
            if peak3 < best_peak:
                best_peak = peak3
                best_A = A3_sa
    
    # Seed 4: Interval-based
    for k in [20, 30, 40]:
        A4 = set(range(0, k + 1)) | set(range(N - k, N + 1))
        A4 = ensure_valid(A4)
        mults4 = get_mults(A4)
        if np.min(mults4) > 0:
            A4_sa, peak4 = simulated_annealing(A4, max_iters=1500)
            if peak4 < best_peak:
                best_peak = peak4
                best_A = A4_sa
    
    # Final local search polish on best
    if best_A is not None:
        # Do a final pass of greedy removal
        A_final = set(best_A)
        improved = True
        while improved and time.time() < deadline - 2:
            improved = False
            mults = get_mults(A_final)
            for a in list(A_final):
                new_mults = mults.copy()
                for i, t in enumerate(TARGETS):
                    b = t - a
                    if 0 <= b <= N and b in A_final:
                        new_mults[i] -= 1
                if np.min(new_mults) == 0:
                    continue
                if np.max(new_mults) < np.max(mults):
                    A_final.remove(a)
                    improved = True
                    break
            best_A = A_final
            best_peak = np.max(get_mults(A_final))
    
    if best_A is None:
        best_A = set(range(64, 129))
    
    return list(best_A)
