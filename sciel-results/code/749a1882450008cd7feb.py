"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 12

# EVOLVE-BLOCK-START
import time
import random
from collections import deque

def _fast_bfs_reset_length(a_trans, b_trans):
    """Computes the exact shortest reset word length using BFS with efficient subset transitions."""
    n = len(a_trans)
    start = tuple(range(n))
    
    # Quick check: if either letter is constant, reset length is 0
    if len(set(a_trans)) == 1 or len(set(b_trans)) == 1:
        return 0
    
    # Precompute transitions for single states to speed up subset transitions
    a_single = [tuple([a_trans[i]]) for i in range(n)]
    b_single = [tuple([b_trans[i]]) for i in range(n)]
    
    visited = {start}
    queue = deque([(start, 0)])
    
    while queue:
        state, dist = queue.popleft()
        if len(state) == 1:
            return dist
        
        # Apply 'a'
        if len(state) == 1:
            next_a = a_single[state[0]]
        else:
            next_a = tuple(sorted(set(a_trans[i] for i in state)))
        if next_a not in visited:
            visited.add(next_a)
            queue.append((next_a, dist + 1))
        
        # Apply 'b'
        if len(state) == 1:
            next_b = b_single[state[0]]
        else:
            next_b = tuple(sorted(set(b_trans[i] for i in state)))
        if next_b not in visited:
            visited.add(next_b)
            queue.append((next_b, dist + 1))
    
    return -1  # Not synchronizing

def _evaluate_pair(a_trans, b_trans):
    """Evaluate the reset length of a pair of transition functions."""
    return _fast_bfs_reset_length(a_trans, b_trans)

def _simulated_annealing(a_trans, b_trans, start_time, deadline):
    """Simulated annealing to maximize reset word length."""
    current_a = a_trans[:]
    current_b = b_trans[:]
    current_score = _evaluate_pair(current_a, current_b)
    
    if current_score < 0:
        return a_trans, b_trans, 0
    
    best_a = current_a[:]
    best_b = current_b[:]
    best_score = current_score
    
    # Temperature schedule
    t0 = 5.0
    cooling = 0.995
    temperature = t0
    
    random.seed(42)
    
    while time.time() < deadline:
        # Generate a mutation: change one or two random destinations
        num_mutations = random.randint(1, 2)
        mutations = []
        
        for _ in range(num_mutations):
            # Choose which transition function to mutate
            if random.random() < 0.5:
                idx = random.randint(0, N - 1)
                old_val = current_a[idx]
                new_val = random.randint(0, N - 1)
                if new_val != old_val:
                    current_a[idx] = new_val
                    mutations.append(('a', idx, old_val, new_val))
            else:
                idx = random.randint(0, N - 1)
                old_val = current_b[idx]
                new_val = random.randint(0, N - 1)
                if new_val != old_val:
                    current_b[idx] = new_val
                    mutations.append(('b', idx, old_val, new_val))
        
        if not mutations:
            continue
        
        # Evaluate the mutated state
        new_score = _evaluate_pair(current_a, current_b)
        
        # Accept if better, or with probability based on temperature
        if new_score > current_score:
            current_score = new_score
            if new_score > best_score:
                best_score = new_score
                best_a = current_a[:]
                best_b = current_b[:]
        else:
            # Accept worse solutions with probability exp(-delta/T)
            if new_score >= 0:
                delta = current_score - new_score
                if delta > 0:
                    prob = min(1.0, 2.718281828 ** (-delta / max(temperature, 0.001)))
                    if random.random() < prob:
                        current_score = new_score
                    else:
                        # Reject and revert
                        for func_name, idx, old_val, new_val in mutations:
                            if func_name == 'a':
                                current_a[idx] = old_val
                            else:
                                current_b[idx] = old_val
                        current_score = _evaluate_pair(current_a, current_b)
            else:
                # Not synchronizing - revert
                for func_name, idx, old_val, new_val in mutations:
                    if func_name == 'a':
                        current_a[idx] = old_val
                    else:
                        current_b[idx] = old_val
                current_score = _evaluate_pair(current_a, current_b)
        
        # Cool down
        temperature *= cooling
        
        # If temperature is very low, restart with higher temperature occasionally
        if temperature < 0.1 and random.random() < 0.01:
            temperature = 2.0
    
    return best_a, best_b, best_score

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin before 90s limit
    
    # Start with a known good synchronizing structure
    # Pinwheel: a is cycle, b shifts down
    a_trans = [(i + 1) % N for i in range(N)]
    b_trans = [0 if i == 0 else i - 1 for i in range(N)]
    
    # Try multiple initial configurations and pick the best
    initial_configs = [
        ([(i + 1) % N for i in range(N)], [0 if i == 0 else i - 1 for i in range(N)]),
        ([(i + 1) % N for i in range(N)], [(i + 1) % N for i in range(N)]),
        ([(2 * i) % N for i in range(N)], [(i + 1) % N for i in range(N)]),
        ([(i + 1) % N for i in range(N)], [(i - 1) % N for i in range(N)]),
    ]
    
    best_overall_a = a_trans[:]
    best_overall_b = b_trans[:]
    best_overall_score = 0
    
    # Try each initial config with simulated annealing
    time_per_config = (deadline - start_time) / len(initial_configs)
    
    for a_init, b_init in initial_configs:
        if time.time() > deadline - 5.0:
            break
        
        config_deadline = min(deadline - 2.0, start_time + time_per_config * (initial_configs.index((a_init, b_init)) + 1))
        
        a_result, b_result, score = _simulated_annealing(a_init, b_init, start_time, config_deadline)
        
        if score > best_overall_score:
            best_overall_score = score
            best_overall_a = a_result
            best_overall_b = b_result
    
    # Final local search refinement on the best result
    if time.time() < deadline - 5.0 and best_overall_score > 0:
        current_a = best_overall_a[:]
        current_b = best_overall_b[:]
        current_score = best_overall_score
        
        # Hill climbing with random restarts
        for _ in range(10):
            if time.time() > deadline - 3.0:
                break
            
            # Random restart: perturb current best
            for _ in range(random.randint(3, 8)):
                if random.random() < 0.5:
                    idx = random.randint(0, N - 1)
                    current_a[idx] = random.randint(0, N - 1)
                else:
                    idx = random.randint(0, N - 1)
                    current_b[idx] = random.randint(0, N - 1)
            
            new_score = _evaluate_pair(current_a, current_b)
            
            if new_score >= 0 and new_score > current_score:
                current_score = new_score
                best_overall_a = current_a[:]
                best_overall_b = current_b[:]
            else:
                # Revert to best
                current_a = best_overall_a[:]
                current_b = best_overall_b[:]
                current_score = best_overall_score
    
    # Ensure we return a valid synchronizing automaton
    final_score = _evaluate_pair(best_overall_a, best_overall_b)
    if final_score < 0:
        # Fallback to a known synchronizing structure
        return [[(i + 1) % N for i in range(N)], [0] * N]
    
    return [best_overall_a, best_overall_b]
# EVOLVE-BLOCK-END
