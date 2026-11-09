import time
import random

def construct():
    N = 512
    start_time = time.time()
    time_limit = 55.0  # Safety margin before 60s kill

    # Precompute 4-term APs: for each element x, list of (o1, o2, o3) such that {x,o1,o2,o3} is a 4-AP
    ap_by_elem = [[] for _ in range(N)]
    for a in range(N):
        for d in range(1, (N - a) // 3 + 1):
            a3 = a + 3 * d
            if a3 >= N:
                break
            p0, p1, p2, p3 = a, a + d, a + 2 * d, a + 3 * d
            ap_by_elem[p0].append((p1, p2, p3))
            ap_by_elem[p1].append((p0, p2, p3))
            ap_by_elem[p2].append((p0, p1, p3))
            ap_by_elem[p3].append((p0, p1, p2))

    def can_add(mask, x):
        """Check if x can be added to set represented by mask without creating a 4-AP."""
        for (o1, o2, o3) in ap_by_elem[x]:
            if (mask >> o1) & 1 and (mask >> o2) & 1 and (mask >> o3) & 1:
                return False
        return True

    def count_bits(mask):
        return bin(mask).count('1')

    def greedy_fill(mask, order=None):
        """Greedily add elements from order to mask if they don't create 4-APs."""
        if order is None:
            order = list(range(N))
        for x in order:
            if (mask >> x) & 1:
                continue
            if can_add(mask, x):
                mask |= (1 << x)
        return mask

    def greedy_fill_random(mask, rng=None):
        """Greedy fill with random element order."""
        if rng is None:
            rng = random
        order = list(range(N))
        rng.shuffle(order)
        return greedy_fill(mask, order)

    def polish(mask, size):
        """Try to add any missing element that doesn't create a 4-AP."""
        changed = True
        while changed:
            changed = False
            for x in range(N):
                if not (mask >> x) & 1:
                    if can_add(mask, x):
                        mask |= (1 << x)
                        size += 1
                        changed = True
        return mask, size

    def swap_local_search(mask, size, rng=None, max_iters=50):
        """
        Local search: try removing k elements and refilling greedily.
        Accept if size increases, or with small probability if equal.
        """
        if rng is None:
            rng = random
        improved = True
        iters = 0
        while improved and iters < max_iters and time.time() - start_time < time_limit:
            improved = False
            iters += 1
            in_set = [x for x in range(N) if (mask >> x) & 1]
            if len(in_set) < 3:
                break
            
            for k in [1, 2, 3]:
                if time.time() - start_time > time_limit:
                    break
                for attempt in range(8):
                    if time.time() - start_time > time_limit:
                        break
                    removals = rng.sample(in_set, k)
                    new_mask = mask
                    for x in removals:
                        new_mask &= ~(1 << x)
                    new_mask = greedy_fill_random(new_mask, rng)
                    new_size = count_bits(new_mask)
                    
                    if new_size > size:
                        mask, size = new_mask, new_size
                        improved = True
                        break
                    elif new_size == size and rng.random() < 0.03:
                        mask = new_mask
                        improved = True
                        break
                if improved:
                    break
        return mask, size

    # Simulated annealing for larger perturbations
    def simulated_annealing(mask, size, rng, t_start=1.0, t_end=0.01, alpha=0.995):
        """SA with K-swap perturbations."""
        current_mask = mask
        current_size = size
        T = t_start
        best_mask = mask
        best_size = size
        
        while T > t_end and time.time() - start_time < time_limit:
            in_set = [x for x in range(N) if (current_mask >> x) & 1]
            if len(in_set) < 3:
                break
            
            # Random perturbation: remove k random elements
            k = rng.choice([2, 3, 4, 5])
            if k > len(in_set):
                k = len(in_set)
            removals = rng.sample(in_set, k)
            new_mask = current_mask
            for x in removals:
                new_mask &= ~(1 << x)
            new_mask = greedy_fill_random(new_mask, rng)
            new_size = count_bits(new_mask)
            
            delta = new_size - current_size
            if delta > 0 or rng.random() < pow(2.718281828, delta / T):
                current_mask = new_mask
                current_size = new_size
                if current_size > best_size:
                    best_size = current_size
                    best_mask = current_mask
            
            T *= alpha
        
        return best_mask, best_size

    # Phase 1: Multiple random greedy initializations
    random.seed(42)
    best_mask = 0
    best_size = 0
    initializations = []
    
    for i in range(300):
        if time.time() - start_time > time_limit:
            break
        m = greedy_fill_random(0)
        sz = count_bits(m)
        if sz > best_size:
            best_size = sz
            best_mask = m
        initializations.append((m, sz))
    
    # Sort initializations by size (best first)
    initializations.sort(key=lambda x: -x[1])
    
    # Phase 2: For each top initialization, do local search then SA
    for idx, (init_mask, init_size) in enumerate(initializations[:50]):
        if time.time() - start_time > time_limit:
            break
        
        # Local search
        m, sz = polish(init_mask, init_size)
        m, sz = swap_local_search(m, sz, max_iters=30)
        
        if sz > best_size:
            best_size = sz
            best_mask = m
        
        # Simulated annealing for top solutions
        if idx < 20 and time.time() - start_time < time_limit:
            m2, sz2 = simulated_annealing(m, sz, random, t_start=0.8, t_end=0.02, alpha=0.998)
            if sz2 > best_size:
                best_size = sz2
                best_mask = m2
    
    # Phase 3: Final polish on best
    best_mask, best_size = polish(best_mask, best_size)
    
    # Phase 4: One more round of local search on the final best
    best_mask, best_size = swap_local_search(best_mask, best_size, max_iters=50)
    best_mask, best_size = polish(best_mask, best_size)
    
    result = [x + 1 for x in range(N) if (best_mask >> x) & 1]
    return sorted(result)
