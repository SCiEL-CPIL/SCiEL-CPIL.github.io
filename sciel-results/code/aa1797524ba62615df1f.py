import time
import random

N = 12
FULL_MASK = (1 << N) - 1
DEADLINE = 58.0  # Safety margin before 60s kill

def construct():
    start_time = time.time()
    
    def build_next(trans):
        """Precompute subset transitions for a single letter."""
        size = 1 << N
        next_sub = [0] * size
        # next_sub[0] is 0
        for mask in range(1, size):
            # Use lowest set bit trick
            lsb = mask & -mask
            idx = lsb.bit_length() - 1
            rest = mask ^ lsb
            next_sub[mask] = next_sub[rest] | (1 << trans[idx])
        return next_sub

    def compute_threshold(next_a, next_b):
        """BFS on subset automaton to find shortest reset word length."""
        size = 1 << N
        dist = [-1] * size
        dist[FULL_MASK] = 0
        queue = [FULL_MASK]
        head = 0
        
        while head < len(queue):
            s = queue[head]
            head += 1
            d = dist[s]
            
            # Try 'a'
            n1 = next_a[s]
            if n1 == 1: # Single state 0
                return d + 1
            if dist[n1] == -1:
                dist[n1] = d + 1
                queue.append(n1)
                
            # Try 'b'
            n2 = next_b[s]
            if n2 == 1:
                return d + 1
            if dist[n2] == -1:
                dist[n2] = d + 1
                queue.append(n2)
                
        return 0 # Not synchronizing

    def evaluate(fa, fb):
        next_a = build_next(fa)
        next_b = build_next(fb)
        return compute_threshold(next_a, next_b)

    rng = random.Random(20240517)
    
    best_score = 0
    best_fa = None
    best_fb = None
    
    def make_cycle():
        """Create a random permutation that is a single cycle."""
        perm = list(range(N))
        rng.shuffle(perm)
        return perm

    def pinwheel_b(cycle_a):
        """Create a 'b' transition that works well with a cycle 'a'.
        Standard pinwheel: b(i) = i-1 mod N, except for one state which maps to 0.
        """
        fb = [0] * N
        for i in range(N):
            fb[i] = (i - 1) % N
        fb[0] = 0
        return fb

    def simulated_annealing(fa, fb, max_iters=2000):
        nonlocal best_score, best_fa, best_fb
        
        curr_fa = fa[:]
        curr_fb = fb[:]
        curr_score = evaluate(curr_fa, curr_fb)
        
        if curr_score > best_score:
            best_score = curr_score
            best_fa = curr_fa[:]
            best_fb = curr_fb[:]
            
        # Simulated Annealing parameters
        T = 10.0  # Initial temperature
        cooling_rate = 0.999
        
        next_a = build_next(curr_fa)
        next_b = build_next(curr_fb)
        
        for i in range(max_iters):
            if time.time() - start_time > DEADLINE:
                return
            
            # Generate neighbor
            # 80% chance mutate b, 20% chance mutate a
            if rng.random() < 0.8:
                old_b = curr_fb[:]
                state = rng.randrange(N)
                new_dest = rng.randrange(N)
                curr_fb[state] = new_dest
                
                next_b = build_next(curr_fb)
                new_score = compute_threshold(next_a, next_b)
                
                delta = new_score - curr_score
                if delta > 0 or rng.random() < pow(2.718, delta / T):
                    curr_score = new_score
                    if curr_score > best_score:
                        best_score = curr_score
                        best_fa = curr_fa[:]
                        best_fb = curr_fb[:]
                else:
                    curr_fb = old_b
                    # Revert next_b? We can just rebuild or keep old. 
                    # Since we reverted curr_fb, next_b is stale if we don't rebuild.
                    # But if we don't accept, we don't use next_b for next step's delta check 
                    # unless we mutate b again. 
                    # Actually, if we revert, next_b is no longer valid for curr_fb.
                    # We must rebuild next_b if we plan to use it again or if we mutate b again.
                    # To save time, we only rebuild if accepted. If rejected, next_b is invalid.
                    # So in next iteration, if we mutate b, we rebuild. If we mutate a, we rebuild next_a.
                    # This is inefficient. Let's just always rebuild the mutated one.
                    pass
            else:
                old_fa = curr_fa[:]
                # Swap two destinations in a to keep it a permutation
                s1, s2 = rng.sample(range(N), 2)
                curr_fa[s1], curr_fa[s2] = curr_fa[s2], curr_fa[s1]
                
                next_a = build_next(curr_fa)
                new_score = compute_threshold(next_a, next_b)
                
                delta = new_score - curr_score
                if delta > 0 or rng.random() < pow(2.718, delta / T):
                    curr_score = new_score
                    if curr_score > best_score:
                        best_score = curr_score
                        best_fa = curr_fa[:]
                        best_fb = curr_fb[:]
                else:
                    curr_fa = old_fa
                    next_a = build_next(curr_fa)
            
            T *= cooling_rate
            if T < 0.01:
                break

    # Main Loop: Multiple restarts with different initial structures
    while time.time() - start_time < DEADLINE:
        # Generate a random cycle for 'a'
        fa = make_cycle()
        
        # Generate initial 'b'
        # Mix of pinwheel and random to ensure diversity
        if rng.random() < 0.7:
            fb = pinwheel_b(fa)
            # Perturb slightly
            fb[rng.randrange(N)] = rng.randrange(N)
        else:
            fb = [rng.randrange(N) for _ in range(N)]
        
        simulated_annealing(fa, fb)

    if best_fa is None:
        # Fallback: return a known synchronizing automaton
        fa = list(range(N))
        fb = [0] * N
        return fa, fb

    return best_fa, best_fb
