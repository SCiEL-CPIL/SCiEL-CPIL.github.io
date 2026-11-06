import time
import random

N = 12
FULL_MASK = (1 << N) - 1
DEADLINE = 55.0  # Safety margin before 60s kill

def construct():
    start_time = time.time()
    size = 1 << N
    
    def build_next(trans):
        """Precompute subset transitions for a single letter."""
        next_sub = [0] * size
        for mask in range(1, size):
            lsb = mask & -mask
            idx = lsb.bit_length() - 1
            rest = mask ^ lsb
            next_sub[mask] = next_sub[rest] | (1 << trans[idx])
        return next_sub

    def compute_threshold(next_a, next_b):
        """BFS on subset automaton to find shortest reset word length."""
        dist = [-1] * size
        dist[FULL_MASK] = 0
        queue = [FULL_MASK]
        head = 0
        
        while head < len(queue):
            s = queue[head]
            head += 1
            d = dist[s]
            
            n1 = next_a[s]
            if n1 == 1:
                return d + 1
            if dist[n1] == -1:
                dist[n1] = d + 1
                queue.append(n1)
                
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
    best_fa = list(range(N))
    best_fb = [0] * N

    def make_cycle():
        fa = list(range(N))
        rng.shuffle(fa)
        return fa

    def make_comb(fa):
        """
        Create a 'b' transition that is a near-permutation or comb structure.
        A common high-threshold structure for binary automata is:
        a: cycle
        b: maps most states to i-1 (or similar shift), but one state maps to 0 (or a fixed point)
        to ensure synchronization.
        """
        fb = [0] * N
        # Standard comb: b(i) = i-1 mod N, but b(0)=0.
        # This creates a "chain" 0<-1<-2...<-N-1 with 0 as sink.
        for i in range(N):
            fb[i] = (i - 1) % N
        fb[0] = 0
        
        # Randomize the "shift" direction or starting point slightly to diversify
        # Actually, let's just shuffle the state labels in the comb structure.
        # If we apply a permutation P to states, the structure remains isomorphic.
        perm = list(range(N))
        rng.shuffle(perm)
        inv_perm = [0] * N
        for i, p in enumerate(perm):
            inv_perm[p] = i
            
        # Original comb: c(i) = i-1 if i>0 else 0
        # New comb: fb[i] = perm[ c(inv_perm[i]) ]
        for i in range(N):
            orig = inv_perm[i]
            if orig == 0:
                dest_orig = 0
            else:
                dest_orig = orig - 1
            fb[i] = perm[dest_orig]
            
        return fb

    # Initialize population with diverse cycle+comb structures
    K = 30
    population = []
    
    for _ in range(K):
        fa = make_cycle()
        fb = make_comb(fa)
        score = evaluate(fa, fb)
        if score > best_score:
            best_score = score
            best_fa = fa[:]
            best_fb = fb[:]
        population.append((score, fa[:], fb[:]))
        
    population.sort(key=lambda x: x[0], reverse=True)
    population = population[:K]
    
    deadline = time.time() + DEADLINE
    
    # Genetic Algorithm with structural constraints
    while time.time() < deadline:
        # Pick a parent (tournament selection or just random from top)
        # Random from top half is a good balance
        parent_score, p_fa, p_fb = population[rng.randrange(min(K, len(population)))]
        
        child_fa = p_fa[:]
        child_fb = p_fb[:]
        
        # Mutation Strategy:
        # 1. Keep 'a' as a permutation (swap two images)
        # 2. Mutate 'b' locally (change one destination) or swap
        r = rng.random()
        
        if r < 0.4:
            # Mutate 'b': change one state's destination
            state = rng.randrange(N)
            child_fb[state] = rng.randrange(N)
        elif r < 0.7:
            # Mutate 'b': swap two destinations
            s1, s2 = rng.sample(range(N), 2)
            child_fb[s1], child_fb[s2] = child_fb[s2], child_fb[s1]
        elif r < 0.9:
            # Mutate 'a': swap two destinations (keep permutation)
            s1, s2 = rng.sample(range(N), 2)
            child_fa[s1], child_fa[s2] = child_fa[s2], child_fa[s1]
        else:
            # Larger jump: Re-initialize 'b' as a new comb structure, keep 'a'
            child_fb = make_comb(child_fa)
            # Perturb the comb
            state = rng.randrange(N)
            child_fb[state] = rng.randrange(N)
            
        score = evaluate(child_fa, child_fb)
        
        if score > best_score:
            best_score = score
            best_fa = child_fa[:]
            best_fb = child_fb[:]
            
        # Update population
        population.append((score, child_fa, child_fb))
        
        # Maintain population size
        if len(population) > K:
            population.sort(key=lambda x: x[0], reverse=True)
            population = population[:K]
        else:
            # If we haven't reached K, keep all, but sort occasionally?
            # Actually, we started with K, so it's always K+1 then trimmed.
            pass
            
    return best_fa, best_fb
