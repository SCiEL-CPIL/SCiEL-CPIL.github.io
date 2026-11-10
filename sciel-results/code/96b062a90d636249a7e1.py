import time
import random
import numpy as np

def construct():
    N = 1000
    max_sum = 2 * N
    best_set = []
    best_len = 0
    
    start_time = time.time()
    deadline = start_time + 115  # Safety margin for 150s limit
    
    def make_state():
        return {
            'chosen': [],
            'counts': [0] * (max_sum + 1),
            'in_set': [False] * (N + 1),
            'full_sums': set()  # sums with count == 2
        }
    
    def can_add(state, x):
        counts = state['counts']
        full = state['full_sums']
        chosen = state['chosen']
        # Check 2x
        if counts[2 * x] >= 2:
            return False
        # Check x + y for all y in chosen
        for y in chosen:
            if counts[x + y] >= 2:
                return False
        return True
    
    def do_add(state, x):
        counts = state['counts']
        full = state['full_sums']
        chosen = state['chosen']
        for y in chosen:
            s = x + y
            counts[s] += 1
            if counts[s] == 2:
                full.add(s)
        s2 = 2 * x
        counts[s2] += 1
        if counts[s2] == 2:
            full.add(s2)
        state['in_set'][x] = True
        chosen.append(x)
    
    def do_remove(state, x):
        counts = state['counts']
        full = state['full_sums']
        chosen = state['chosen']
        idx = chosen.index(x)
        for i, y in enumerate(chosen):
            if y == x:
                continue
            s = x + y
            counts[s] -= 1
            if counts[s] == 1:
                full.discard(s)
        s2 = 2 * x
        counts[s2] -= 1
        if counts[s2] == 1:
            full.discard(s2)
        state['in_set'][x] = False
        chosen.pop(idx)
    
    def greedy_build(order, seed=None):
        if seed is not None:
            rng = random.Random(seed)
        else:
            rng = random
            
        if order == 'random':
            candidates = list(range(1, N + 1))
            rng.shuffle(candidates)
        elif order == 'squares':
            squares = [i*i for i in range(1, int(np.sqrt(N)) + 1) if i*i <= N]
            others = [x for x in range(1, N + 1) if x not in squares]
            rng.shuffle(others)
            candidates = squares + others
        elif order == 'ascending':
            candidates = list(range(1, N + 1))
        elif order == 'descending':
            candidates = list(range(N, 0, -1))
        elif order == 'weighted_small':
            candidates = list(range(1, N + 1))
            weights = [1.0 / (1 + i * 0.01) for i in range(N)]
            indexed = list(zip(weights, candidates))
            rng.shuffle(indexed)
            candidates = [x[1] for x in indexed]
        elif order == 'sparse_fill':
            sparse = list(range(1, N + 1, 3))
            fill = list(range(2, N + 1, 3))
            rng.shuffle(fill)
            candidates = sparse + fill
        elif order == 'power2':
            # Powers of 2 first, then fill
            powers = [2**i for i in range(11) if 2**i <= N]
            others = [x for x in range(1, N + 1) if x not in powers]
            rng.shuffle(others)
            candidates = powers + others
        else:
            candidates = list(range(1, N + 1))
            rng.shuffle(candidates)
            
        state = make_state()
        for x in candidates:
            if can_add(state, x):
                do_add(state, x)
        return state
    
    def local_search(state, max_rounds=50, time_budget=10.0, rng=None):
        """Fast local search using full_sums tracking."""
        if rng is None:
            rng = random
        chosen = state['chosen']
        counts = state['counts']
        in_set = state['in_set']
        full = state['full_sums']
        
        t0 = time.time()
        rounds = 0
        improved = True
        
        while improved and rounds < max_rounds:
            if time.time() - t0 > time_budget:
                break
            improved = False
            rounds += 1
            
            # Phase 1: Try to add any element
            candidates = [x for x in range(1, N + 1) if not in_set[x]]
            rng.shuffle(candidates)
            for x in candidates:
                if time.time() - t0 > time_budget:
                    break
                if can_add(state, x):
                    do_add(state, x)
                    improved = True
                    break
            
            if improved:
                continue
                
            # Phase 2: Try swaps - remove x, try to add y
            if len(chosen) == 0:
                break
                
            # Sample a few elements to try removing
            n_tries = min(20, len(chosen))
            indices = rng.sample(range(len(chosen)), n_tries)
            
            for idx in indices:
                if time.time() - t0 > time_budget:
                    break
                x = chosen[idx]
                
                # Temporarily remove x
                do_remove(state, x)
                
                # Try to add a new element
                added_y = -1
                candidates = [y for y in range(1, N + 1) if not in_set[y]]
                rng.shuffle(candidates)
                for y in candidates:
                    if time.time() - t0 > time_budget:
                        break
                    if can_add(state, y):
                        do_add(state, y)
                        added_y = y
                        break
                
                if added_y == -1:
                    # Put x back
                    do_add(state, x)
                else:
                    improved = True
                    break
                    
        return state
    
    def get_size(state):
        return len(state['chosen'])
    
    def update_best(state):
        nonlocal best_set, best_len
        s = get_size(state)
        if s > best_len:
            best_len = s
            best_set = state['chosen'][:]
    
    # Strategy 1: Squares initialization (often good for B2 sets)
    if time.time() < deadline - 5:
        state = greedy_build('squares', 42)
        state = local_search(state, max_rounds=100, time_budget=min(20, deadline - time.time() - 2), rng=random.Random(42))
        update_best(state)
    
    # Strategy 2: Ascending greedy
    if time.time() < deadline - 5:
        state = greedy_build('ascending', 42)
        state = local_search(state, max_rounds=80, time_budget=min(15, deadline - time.time() - 2), rng=random.Random(43))
        update_best(state)
    
    # Strategy 3: Descending greedy
    if time.time() < deadline - 5:
        state = greedy_build('descending', 42)
        state = local_search(state, max_rounds=80, time_budget=min(15, deadline - time.time() - 2), rng=random.Random(44))
        update_best(state)
    
    # Strategy 4: Power of 2 initialization
    if time.time() < deadline - 5:
        state = greedy_build('power2', 42)
        state = local_search(state, max_rounds=80, time_budget=min(15, deadline - time.time() - 2), rng=random.Random(45))
        update_best(state)
    
    # Strategy 5: Weighted small numbers
    if time.time() < deadline - 5:
        state = greedy_build('weighted_small', 42)
        state = local_search(state, max_rounds=80, time_budget=min(15, deadline - time.time() - 2), rng=random.Random(46))
        update_best(state)
    
    # Strategy 6: Sparse fill
    if time.time() < deadline - 5:
        state = greedy_build('sparse_fill', 42)
        state = local_search(state, max_rounds=80, time_budget=min(15, deadline - time.time() - 2), rng=random.Random(47))
        update_best(state)
    
    # Strategy 7: Random restarts with diverse orderings
    rng = random.Random(123)
    orders = ['random', 'random', 'random', 'weighted_small', 'sparse_fill', 'power2', 'squares']
    while time.time() < deadline - 3:
        order_choice = rng.choice(orders)
        seed = rng.randint(0, 1000000)
        state = greedy_build(order_choice, seed)
        remaining = deadline - time.time() - 3
        if remaining < 2:
            break
        state = local_search(state, max_rounds=40, time_budget=min(8, remaining), rng=random.Random(seed))
        update_best(state)
    
    # Final polish: try to add more to best set
    if best_set and time.time() < deadline - 1:
        state = make_state()
        for x in best_set:
            do_add(state, x)
        state = local_search(state, max_rounds=30, time_budget=min(5, deadline - time.time() - 1), rng=random.Random(999))
        update_best(state)
    
    return best_set
