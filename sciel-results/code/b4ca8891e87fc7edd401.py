import numpy as np
import random
import time

N = 18

def construct():
    start_time = time.time()
    timeout = 55
    
    # Precompute all possible comparators
    all_comps = [(i, j) for i in range(N) for j in range(i+1, N)]
    
    # Sample inputs for quick evaluation
    # Use a fixed set of "hard" inputs: all 0s, all 1s, alternating, etc.
    # And some random ones.
    sample_inputs = []
    # Add specific patterns
    sample_inputs.append([0]*N)
    sample_inputs.append([1]*N)
    sample_inputs.append([i%2 for i in range(N)])
    sample_inputs.append([i%2 for i in range(N-1, -1, -1)])
    for _ in range(20):
        sample_inputs.append([random.randint(0, 1) for _ in range(N)])
    
    def apply_network(state, comparators):
        state = list(state)
        for i, j in comparators:
            if state[i] > state[j]:
                state[i], state[j] = state[j], state[i]
        return state

    def count_inversions(state):
        inv = 0
        for i in range(N):
            for j in range(i+1, N):
                if state[i] > state[j]:
                    inv += 1
        return inv

    def score_network(comparators):
        # Returns (num_comparators, total_inversions_on_samples)
        inv_sum = 0
        for inp in sample_inputs:
            out = apply_network(inp, comparators)
            inv_sum += count_inversions(out)
        return (len(comparators), inv_sum)

    def is_valid(comparators):
        # Full check
        for mask in range(1 << N):
            state = [(mask >> i) & 1 for i in range(N)]
            for i, j in comparators:
                if state[i] > state[j]:
                    state[i], state[j] = state[j], state[i]
            # Check sorted
            for k in range(N - 1):
                if state[k] > state[k + 1]:
                    return False
        return True

    # Beam Search
    beam_size = 50
    max_depth = 12
    
    # Initial beam: empty network
    beam = [[]]
    
    best_network = None
    best_score = float('inf')
    
    for depth in range(max_depth):
        if time.time() - start_time > timeout:
            break
            
        new_beam = []
        for net in beam:
            # Generate candidate next layers
            # 1. Empty layer
            # 2. Random matchings
            # 3. Greedy matching based on current state disorder?
            #    To keep it simple, we generate random matchings.
            
            candidates = [net] # Empty layer
            
            # Generate 5 random matchings
            for _ in range(5):
                # Generate a random matching (set of non-conflicting comparators)
                used = [False] * N
                matching = []
                # Shuffle wires to randomize
                wires = list(range(N))
                random.shuffle(wires)
                for i in wires:
                    if used[i]:
                        continue
                    # Find a random available j
                    available = [j for j in wires if not used[j] and j != i]
                    if not available:
                        break
                    j = random.choice(available)
                    if i < j:
                        matching.append((i, j))
                    else:
                        matching.append((j, i))
                    used[i] = True
                    used[j] = True
                candidates.append(net + matching)
            
            # Evaluate and add to new beam
            for cand in candidates:
                if time.time() - start_time > timeout:
                    break
                score = score_network(cand)
                new_beam.append((score, cand))
        
        # Sort new beam by score (comparator count, then inversions)
        new_beam.sort(key=lambda x: x[0])
        
        # Keep top beam_size
        beam = [item[1] for item in new_beam[:beam_size]]
        
        # Check if any in beam is valid
        # We only check the top few to save time
        for net in beam[:5]:
            if is_valid(net):
                best_network = net
                best_score = len(net)
                # Try to improve by increasing depth? 
                # Or just break and do local search?
                # The directive says "SAT-inspired". 
                # If we find a valid one, we can try to reduce size by 
                # continuing the search with a bias towards fewer comparators?
                # Actually, if we found a valid one at depth D, 
                # we can try to see if a shorter one exists?
                # No, we are building up. 
                # Let's try to find a valid one with fewer comparators 
                # by restarting with a smaller beam or different random seed?
                
                # For now, let's accept this and try to polish it 
                # with a few steps of local search (removing redundant comps).
                # This is a hybrid, but the core is the Beam Search construction.
                break
        
        if best_network:
            # Try to reduce size with a quick local search
            # This part is similar to the old program, but the initial solution 
            # is different (constructed via beam search vs seed).
            current = best_network
            for _ in range(50): # 50 iterations of local search
                if time.time() - start_time > timeout:
                    break
                improved = False
                indices = list(range(len(current)))
                random.shuffle(indices)
                for idx in indices:
                    if time.time() - start_time > timeout:
                        break
                    new_comps = current[:idx] + current[idx+1:]
                    if is_valid(new_comps):
                        current = new_comps
                        improved = True
                        if len(current) < best_score:
                            best_network = current
                            best_score = len(current)
                if not improved:
                    break
            
            return best_network
            
    # If no valid network found, return a known good one (Odd-Even)
    # This should not happen if depth is sufficient
    comps = []
    for phase in range(N):
        for i in range(phase % 2, N - 1, 2):
            comps.append((i, i + 1))
    return comps
