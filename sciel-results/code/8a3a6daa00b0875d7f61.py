import random
import time
import math

N = 256

def construct():
    start_time = time.time()
    deadline = start_time + 55.0

    # Precompute 3-APs and 4-APs
    threes = []
    for d in range(1, N // 2 + 1):
        for a in range(0, N - 2 * d):
            threes.append((a, a + d, a + 2 * d))

    fours = []
    for d in range(1, N // 3 + 1):
        for a in range(0, N - 3 * d):
            fours.append((a, a + d, a + 2 * d, a + 3 * d))

    # Precompute index -> list of (other_indices) for fast delta
    index_3aps = [[] for _ in range(N)]
    for a, b, c in threes:
        index_3aps[a].append((b, c))
        index_3aps[b].append((a, c))
        index_3aps[c].append((a, b))

    index_4aps = [[] for _ in range(N)]
    for a, b, c, d in fours:
        index_4aps[a].append((b, c, d))
        index_4aps[b].append((a, c, d))
        index_4aps[c].append((a, b, d))
        index_4aps[d].append((a, b, c))

    def calc_scores(state):
        s3 = 0
        s4 = 0
        for a, b, c in threes:
            if state[a] and state[b] and state[c]:
                s3 += 1
        for a, b, c, d in fours:
            if state[a] and state[b] and state[c] and state[d]:
                s4 += 1
        return s3, s4

    def get_delta(state, x, new_val):
        """
        Calculate (delta_s3, delta_s4) if state[x] is set to new_val.
        Assumes state[x] is currently (not new_val).
        """
        delta_s3 = 0
        delta_s4 = 0
        sign = 1 if new_val else -1

        for o1, o2 in index_3aps[x]:
            if state[o1] and state[o2]:
                delta_s3 += sign

        for o1, o2, o3 in index_4aps[x]:
            if state[o1] and state[o2] and state[o3]:
                delta_s4 += sign

        return delta_s3, delta_s4

    def init_greedy():
        in_set = [False] * N
        # Greedy: add x if it doesn't create a 4-AP
        # Order: random shuffle to break symmetry
        order = list(range(N))
        random.shuffle(order)
        
        for x in order:
            creates_4ap = False
            for o1, o2, o3 in index_4aps[x]:
                if in_set[o1] and in_set[o2] and in_set[o3]:
                    creates_4ap = True
                    break
            if not creates_4ap:
                in_set[x] = True
        return in_set

    def perturb(state, k):
        new_state = state[:]
        for _ in range(k):
            idx = random.randint(0, N - 1)
            new_state[idx] = not new_state[idx]
        return new_state

    def hill_climb(state, s3, s4, time_limit):
        # Strict hill climbing with random restarts/perturbations if stuck
        # Actually, let's do a simple local search: try flipping bits that improve score
        # Score = s3 - 100 * s4
        
        penalty = 100.0
        current_score = s3 - penalty * s4
        
        improved = True
        while improved and time.time() < time_limit:
            improved = False
            # Try all possible flips
            # To speed up, we can shuffle the order
            order = list(range(N))
            random.shuffle(order)
            
            for x in order:
                new_val = not state[x]
                ds3, ds4 = get_delta(state, x, new_val)
                new_score = (s3 + ds3) - penalty * (s4 + ds4)
                
                if new_score > current_score:
                    state[x] = new_val
                    s3 += ds3
                    s4 += ds4
                    current_score = new_score
                    improved = True
        
        return state, s3, s4

    def simulated_annealing(state, s3, s4, T_start, T_end, steps, time_limit):
        penalty = 100.0
        current_score = s3 - penalty * s4
        
        T = T_start
        cooling = (T_end / T_start) ** (1.0 / max(1, steps))
        
        for _ in range(steps):
            if time.time() > time_limit:
                break
                
            x = random.randint(0, N - 1)
            new_val = not state[x]
            ds3, ds4 = get_delta(state, x, new_val)
            new_score = (s3 + ds3) - penalty * (s4 + ds4)
            delta_e = new_score - current_score
            
            if delta_e > 0 or random.random() < math.exp(delta_e / T):
                state[x] = new_val
                s3 += ds3
                s4 += ds4
                current_score = new_score
            
            T *= cooling
            
        return state, s3, s4

    best_valid_set = None
    best_valid_score = -1

    # Run multiple iterations of: Init -> SA -> Hill Climb
    # We have ~50 seconds. Let's try to fit as many as possible.
    # Each iteration: SA (maybe 2000 steps) + HC (variable)
    
    iterations = 0
    while time.time() < deadline:
        # 1. Initialize
        if iterations == 0:
            state = init_greedy()
        else:
            # Perturb best found so far or re-initialize?
            # Let's mix: 50% perturb best, 50% new greedy
            if best_valid_set is not None and random.random() < 0.5:
                state = perturb(best_valid_set, random.randint(5, 20))
            else:
                state = init_greedy()
        
        s3, s4 = calc_scores(state)
        
        # 2. Simulated Annealing
        # Time budget for SA: 5s per iteration?
        # Let's do fixed steps but check time
        steps = 3000
        state, s3, s4 = simulated_annealing(state, s3, s4, 2.0, 0.01, steps, deadline)
        
        # 3. Hill Climb
        # Use remaining time up to 2s per iteration?
        hc_deadline = min(deadline, time.time() + 2.0)
        state, s3, s4 = hill_climb(state, s3, s4, hc_deadline)
        
        # 4. Update best
        if s4 == 0 and s3 > best_valid_score:
            best_valid_score = s3
            best_valid_set = state[:]
            
        iterations += 1
        # Small break to ensure we don't run over
        if time.time() > deadline - 5.0:
            break

    # If no valid set found (shouldn't happen with greedy), return greedy
    if best_valid_set is None:
        state = init_greedy()
        best_valid_set = state

    # Convert to 1-based list
    result = [i + 1 for i in range(N) if best_valid_set[i]]
    return result

# Fallback if construct fails? No, just return the function.
# The task requires construct() to be the entry point.
# I will expose construct as defined.

# For testing purposes, if run directly
if __name__ == "__main__":
    res = construct()
    print(f"Size: {len(res)}")
    # Verify
    in_set = [False] * N
    for x in res:
        in_set[x-1] = True
    s3, s4 = calc_scores(in_set)
    print(f"3-APs: {s3}, 4-APs: {s4}")
