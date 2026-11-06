import random
import time

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s margin for safety
    
    # Prescribed elements
    initial = [1, 2, 5, 11]
    best_size = len(initial)
    best_set = initial[:]
    
    # Pre-calculate sums for initial set
    def get_sums(s):
        sums = set()
        n = len(s)
        for i in range(n):
            for j in range(i, n):
                sums.add(s[i] + s[j])
        return sums
    
    best_sums = get_sums(best_set)
    
    # Candidates are 12 to 284
    candidates = list(range(12, 285))
    
    def is_compatible(x, current_set, current_sums):
        # Check if x can be added
        if 2 * x in current_sums:
            return False
        for y in current_set:
            if x + y in current_sums:
                return False
        return True
        
    def greedy_fill(current_set, current_sums, candidates_pool, priority_mode='random'):
        """
        Greedily add elements from a pool until no more can be added.
        priority_mode: 'random', 'gap', or 'compatibility'
        """
        s = current_set[:]
        sums = set(current_sums)
        
        if priority_mode == 'random':
            pool = candidates_pool[:]
            random.shuffle(pool)
        elif priority_mode == 'gap':
            # Sort by gap coverage - prefer elements that fill large gaps
            pool = candidates_pool[:]
            # Score each candidate by how well it fits in gaps
            def gap_score(x):
                # Find nearest elements in s
                dists = []
                for y in s:
                    dists.append(abs(x - y))
                if not dists:
                    return 0
                # Prefer elements that are far from existing ones (fill gaps)
                return max(dists)
            pool.sort(key=gap_score, reverse=True)
        elif priority_mode == 'compatibility':
            # Score by how many valid pairs x can form with existing elements
            pool = candidates_pool[:]
            def compat_score(x):
                count = 0
                for y in s:
                    if (x + y) not in sums and (2*x) not in sums:
                        count += 1
                return count
            pool.sort(key=compat_score, reverse=True)
        else:
            pool = candidates_pool[:]
            random.shuffle(pool)
        
        for x in pool:
            if is_compatible(x, s, sums):
                s.append(x)
                for y in s[:-1]:
                    sums.add(x + y)
                sums.add(2 * x)
        
        return s, sums

    def local_search(start_set, start_sums):
        """
        Local search with multi-kick operations.
        """
        s = start_set[:]
        sums = set(start_sums)
        current_size = len(s)
        
        # First, try to fill greedily with different strategies
        for mode in ['random', 'gap', 'compatibility']:
            s_try, sums_try = greedy_fill(s, sums, candidates, mode)
            if len(s_try) > current_size:
                s, sums, current_size = s_try, sums_try, len(s_try)
        
        # Multi-kick: remove 1-3 elements and refill
        for kick_size in [1, 2, 3]:
            for _ in range(5):  # Try a few kicks of each size
                if time.time() > deadline - 5:
                    break
                removable_indices = [i for i, x in enumerate(s) if x not in initial]
                if len(removable_indices) < kick_size:
                    break
                    
                # Remove kick_size random elements
                to_remove = random.sample(removable_indices, kick_size)
                for idx in sorted(to_remove, reverse=True):
                    s.pop(idx)
                sums = get_sums(s)
                
                # Refill with random greedy
                s_try, sums_try = greedy_fill(s, sums, candidates, 'random')
                
                if len(s_try) > current_size:
                    s, sums, current_size = s_try, sums_try, len(s_try)
        
        return s, sums, current_size

    # Main search loop
    search_strategies = [
        ('random', None),
        ('gap', None),
        ('compatibility', None),
        ('random', 'local'),
        ('gap', 'local'),
    ]
    
    strategy_idx = 0
    while time.time() < deadline:
        mode = search_strategies[strategy_idx % len(search_strategies)][0]
        use_local = search_strategies[strategy_idx % len(search_strategies)][1]
        strategy_idx += 1
        
        s = initial[:]
        sums = get_sums(s)
        
        # Greedy fill
        s, sums = greedy_fill(s, sums, candidates, mode)
        
        # Local search if requested
        if use_local:
            s, sums, _ = local_search(s, sums)
        
        if len(s) > best_size:
            best_size = len(s)
            best_set = s[:]
            best_sums = set(sums)
            
            if best_size >= 20:
                return best_set
        
        # Simple kick for diversity
        if best_size < 20 and len(s) > 4:
            removable_indices = [i for i, x in enumerate(s) if x not in initial]
            if removable_indices:
                idx = random.choice(removable_indices)
                s.pop(idx)
                sums = get_sums(s)
                s, sums = greedy_fill(s, sums, candidates, 'random')
                
                if len(s) > best_size:
                    best_size = len(s)
                    best_set = s[:]
                    best_sums = set(sums)
                    if best_size >= 20:
                        return best_set

    return best_set
