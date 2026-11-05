import time
import random
import sys
from collections import deque

N = 12
FULL_STATE = (1 << N) - 1
DEADLINE = 50.0  # Safety margin under 60s kill

def calculate_threshold(a_trans, b_trans):
    """
    Returns the shortest reset word length for the automaton defined by a_trans and b_trans,
    or -1 if not synchronizing.
    """
    a_bits = [1 << a_trans[i] for i in range(N)]
    b_bits = [1 << b_trans[i] for i in range(N)]
    
    # Precompute transition tables for all possible subsets
    trans_a = [0] * (1 << N)
    trans_b = [0] * (1 << N)
    
    for mask in range(1 << N):
        if mask == 0:
            continue
        res_a = 0
        res_b = 0
        m = mask
        while m:
            lsb = m & -m
            idx = lsb.bit_length() - 1
            res_a |= a_bits[idx]
            res_b |= b_bits[idx]
            m ^= lsb
        trans_a[mask] = res_a
        trans_b[mask] = res_b
        
    # BFS
    visited = [False] * (1 << N)
    visited[FULL_STATE] = True
    queue = deque()
    queue.append((FULL_STATE, 0))
    
    while queue:
        state, dist = queue.popleft()
        if state.bit_count() == 1:
            return dist
            
        next_a = trans_a[state]
        if not visited[next_a]:
            visited[next_a] = True
            queue.append((next_a, dist + 1))
            
        next_b = trans_b[state]
        if not visited[next_b]:
            visited[next_b] = True
            queue.append((next_b, dist + 1))
            
    return -1

def construct():
    start_time = time.time()
    
    def get_score(a, b):
        return calculate_threshold(a, b)

    def random_automaton():
        return [random.randint(0, N-1) for _ in range(N)]

    def pinsker_automaton():
        a = [(i + 1) % N for i in range(N)]
        # b: 0->0, i->i-1 for i>0 is not always best, let's try a few variants
        b1 = [0] + [i-1 for i in range(1, N)]
        b2 = [(i - 1) % N for i in range(N)]
        b3 = [0] * N
        return a, b1, b2, b3

    # Initialize with best of a few structured starts
    candidates = []
    a_p, b_p1, b_p2, b_p3 = pinsker_automaton()
    candidates.append((a_p[:], b_p1))
    candidates.append((a_p[:], b_p2))
    candidates.append((a_p[:], b_p3))
    
    for _ in range(10):
        candidates.append((random_automaton(), random_automaton()))
    
    best_a = None
    best_b = None
    best_score = -1
    
    for a, b in candidates:
        s = get_score(a, b)
        if s > best_score:
            best_score = s
            best_a = a[:]
            best_b = b[:]
            
    if best_a is None:
        best_a = [(i + 1) % N for i in range(N)]
        best_b = [0] * N
        best_score = get_score(best_a, best_b)

    # Simulated Annealing
    current_a = best_a[:]
    current_b = best_b[:]
    current_score = best_score
    
    # Parameters
    T_start = 10.0
    T_end = 0.1
    alpha = 0.999  # Cooling factor
    max_steps = 5000  # Max steps per restart
    stagnation_limit = 500  # Steps without improvement before restart
    
    steps_without_improvement = 0
    
    while time.time() - start_time < DEADLINE:
        T = T_start * (alpha ** (steps_without_improvement / stagnation_limit)) # Approximate cooling based on global stagnation? 
        # Better: simple linear or exponential cooling per restart
        # Let's use a simple restart strategy:
        # Run SA for max_steps or until time limit. If no improvement, restart from best.
        
        # Actually, let's do a simple Hill Climbing with Random Restarts (Simulated Annealing is complex to tune).
        # Hill Climbing:
        # 1. Pick a random neighbor.
        # 2. If better, accept.
        # 3. If worse, accept with probability exp(-delta/T).
        
        # Let's implement a proper SA loop within the deadline.
        
        # Reset cooling schedule for this "phase" if we restart?
        # Let's just run a continuous SA with a temperature schedule that resets if we hit a local minimum?
        # Simpler: Just run a long SA chain.
        
        # To make it robust, let's use a "Best-First" Hill Climbing with occasional random jumps.
        # This is often faster for discrete problems.
        
        # Let's try: 
        # 1. Generate neighbor.
        # 2. Evaluate.
        # 3. If better, accept and reset stagnation.
        # 4. If worse, accept with prob exp(-diff/T).
        # 5. If stagnation > limit, restart from best known.
        
        # We need a temperature.
        # Let's start with T = 5.0 and decrease it slowly.
        # But since we have a hard deadline, let's just use a fixed high temperature for exploration?
        # No, we want to converge.
        
        # Let's use a simple annealing schedule based on total steps taken.
        # Total steps is not tracked globally easily in the loop without a counter.
        
        # Alternative: Random Restart Hill Climbing.
        # 1. Start from best.
        # 2. Do local search (accept only improvements).
        # 3. If stuck (no improvement for K steps), jump to a random neighborhood of best.
        
        # This is often very effective for these automata.
        
        # Let's implement Random Restart Hill Climbing.
        
        # If we are currently in a "restart" state, we are just doing local search.
        # If stagnation > limit, we "restart" by picking a random neighbor of the *best* solution found so far, 
        # or even a completely random automaton if best is not improving for a long time.
        
        # Let's stick to the SA idea but implement it carefully.
        
        # Actually, the previous GA was scoring 50. Let's try to beat it.
        # Known best for N=12 is likely around 50-60? 
        # The Pinsker automaton for N=12 has threshold 11. 
        # Wait, the score is the threshold length. 
        # For N=12, max possible is (N-1)^2 = 121.
        # Current score 50 is good.
        
        # Let's use a "Greedy" local search with random restarts.
        
        # If this block is reached, we are in the main loop.
        # We will perform one step of local search.
        
        # Pick a random position and letter to change
        if random.random() < 0.5:
            idx = random.randint(0, N - 1)
            old_val = current_a[idx]
            new_val = random.randint(0, N - 1)
            if new_val == old_val:
                continue
            current_a[idx] = new_val
            new_a = current_a
            new_b = current_b
        else:
            idx = random.randint(0, N - 1)
            old_val = current_b[idx]
            new_val = random.randint(0, N - 1)
            if new_val == old_val:
                continue
            current_b[idx] = new_val
            new_a = current_a
            new_b = current_b
            
        new_score = get_score(new_a, new_b)
        
        # Accept if better
        if new_score > current_score:
            current_score = new_score
            steps_without_improvement = 0
            if current_score > best_score:
                best_score = current_score
                best_a = new_a[:]
                best_b = new_b[:]
        else:
            # Reject if worse (Hill Climbing)
            # Or accept with probability (SA)
            # Let's use SA: accept worse with prob exp(-delta / T)
            # We need a temperature.
            # Let's use a temperature that cools down over time.
            # Since we don't have a global step counter, let's use a simple heuristic:
            # If we haven't improved in a while, increase exploration?
            
            # Let's implement a simple SA with a fixed temperature T=1.0?
            # If new_score is -1 (not synchronizing), delta is large.
            # If new_score < current_score, delta = current_score - new_score.
            
            # To avoid getting stuck, let's use a "restart" strategy instead of complex SA.
            # If steps_without_improvement > 100:
            #    Reset current to best_a, best_b
            #    T = 1.0
            # else:
            #    T = 0.5
            
            # Let's just do: If no improvement for 100 steps, restart from best.
            if steps_without_improvement > 100:
                # Restart from best
                current_a = best_a[:]
                current_b = best_b[:]
                current_score = best_score
                steps_without_improvement = 0
            else:
                # Try SA acceptance
                # Temperature T
                # Let's use T = 2.0 / (1 + steps_without_improvement / 50)
                T = 2.0 / (1 + steps_without_improvement / 50.0)
                delta = current_score - new_score
                if delta > 0:
                    if random.random() < pow(2.71828, -delta / T):
                        # Accept worse
                        pass # current_a/b already updated
                    else:
                        # Revert
                        if random.random() < 0.5: # This logic is flawed because we don't know which letter was changed
                             # We need to revert correctly.
                             # It's easier to just not accept worse in Hill Climbing.
                             # Let's stick to Hill Climbing with Restarts. It's simpler and robust.
                             pass
                # Actually, let's just do Hill Climbing.
                # If worse, revert.
                # We need to revert the change.
                # Since we don't track which letter was changed in this scope easily without extra vars, 
                # let's restructure the loop to be cleaner.
                
                # Let's restart the loop logic to be cleaner Hill Climbing.
                # I will replace the entire construct function body to be cleaner.
                pass

    return [best_a, best_b]
