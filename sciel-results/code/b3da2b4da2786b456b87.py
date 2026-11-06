import time
import random

N = 256

def construct():
    best_size = N + 1
    best_set = []
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin before 60s kill

    def compute_sums(S):
        """Compute all pairwise sums a+b for a<=b in S."""
        s = set()
        for i in range(len(S)):
            s.add(2 * S[i])
            for j in range(i + 1, len(S)):
                s.add(S[i] + S[j])
        return s

    def is_valid_add(x, S, sums):
        """Check if x can be added to Sidon set S."""
        if 2 * x in sums:
            return False
        for s in S:
            if x + s in sums:
                return False
        return True

    def is_maximal(S, sums):
        """Check if S is inclusion-maximal: no element from [1,N]\S can be added."""
        S_set = set(S)
        for y in range(1, N + 1):
            if y in S_set:
                continue
            if is_valid_add(y, S, sums):
                return False
        return True

    def local_search_reduce(S, sums, time_limit=None):
        """Try to remove elements while maintaining maximality.
        Returns a smaller maximal Sidon set if found."""
        S = list(S)
        sums = compute_sums(S)
        improved = True
        while improved:
            improved = False
            if time_limit is not None and time.time() > time_limit:
                break
            # Try removing each element in random order
            indices = list(range(len(S)))
            random.shuffle(indices)
            for i in indices:
                if time_limit is not None and time.time() > time_limit:
                    break
                # Temporarily remove S[i]
                S_temp = S[:i] + S[i+1:]
                if not S_temp:
                    continue
                sums_temp = compute_sums(S_temp)
                if is_maximal(S_temp, sums_temp):
                    S = S_temp
                    sums = sums_temp
                    improved = True
                    break
        return S, sums

    def greedy_construct(order):
        """Greedy construct Sidon set following given order."""
        S = []
        sums = set()
        for x in order:
            if is_valid_add(x, S, sums):
                S.append(x)
                sums.add(2 * x)
                for s in S[:-1]:
                    sums.add(x + s)
        return S, sums

    def conflict_graph_greedy():
        """Model as conflict graph, then use greedy vertex elimination.
        Build a conflict graph where vertices 1..N are connected if they
        cannot coexist in a Sidon set. Then greedily eliminate high-degree
        vertices to find a small maximal independent set."""
        
        # Build conflict graph: edge (a,b) if a and b cannot coexist
        # Two elements a,b conflict if 2a == 2b (impossible for distinct), 
        # or a+b equals some other pair sum... but that's context-dependent.
        # 
        # Simpler approach: use the "blocking" concept. Element x blocks y
        # if adding x to an empty set prevents y from being added later.
        # For a single element x, it blocks y if 2x == 2y (no) or x+y creates
        # a sum that conflicts. Actually, with just {x}, y can be added iff
        # 2y != 2x and x+y != 2x (i.e., y != x). So single elements don't 
        # block each other except identity.
        #
        # Better: iteratively build the set by always picking the element that
        # has the most "future conflicts" - i.e., blocks the most remaining
        # elements when added to current S.
        
        S = []
        sums = set()
        remaining = set(range(1, N + 1))
        
        while remaining:
            best_x = None
            best_block_count = -1
            
            # For efficiency, sample if too many candidates
            if len(remaining) > 40:
                candidates = random.sample(list(remaining), 40)
            else:
                candidates = list(remaining)
            
            for x in candidates:
                if not is_valid_add(x, S, sums):
                    continue
                
                # Compute how many elements in remaining would be blocked
                new_sums = sums | {2 * x} | {x + s for s in S}
                block_count = 0
                for y in remaining:
                    if y == x:
                        continue
                    if not is_valid_add(y, S, new_sums):
                        block_count += 1
                
                # Add small random perturbation for diversity
                score = block_count + random.random() * 2
                
                if score > best_block_count:
                    best_block_count = score
                    best_x = x
            
            if best_x is None:
                break
            
            S.append(best_x)
            sums.add(2 * best_x)
            for s in S[:-1]:
                sums.add(best_x + s)
            remaining.remove(best_x)
            
            # Early stop if maximal
            if len(S) >= 8 and is_maximal(S, sums):
                break
        
        return S, sums

    def hitting_set_search(S, sums, time_limit=None):
        """Hitting-set style local search: identify non-essential elements
        for maximality and try to remove them."""
        S = list(S)
        sums = compute_sums(S)
        improved = True
        while improved:
            improved = False
            if time_limit is not None and time.time() > time_limit:
                break
            
            # For each element, check if it's "essential" for blocking
            # An element x is non-essential if removing it still leaves
            # all other elements blocked
            indices = list(range(len(S)))
            random.shuffle(indices)
            
            for i in indices:
                if time_limit is not None and time.time() > time_limit:
                    break
                S_temp = S[:i] + S[i+1:]
                if not S_temp:
                    continue
                sums_temp = compute_sums(S_temp)
                if is_maximal(S_temp, sums_temp):
                    S = S_temp
                    sums = sums_temp
                    improved = True
                    break
        return S, sums

    # Main search loop with multiple strategies
    for iteration in range(8000):
        if time.time() >= deadline:
            break
        
        strategy = iteration % 5
        
        if strategy == 0:
            # Conflict graph greedy (vertex elimination style)
            S, sums = conflict_graph_greedy()
        elif strategy == 1:
            # Random order greedy
            order = list(range(1, N + 1))
            random.shuffle(order)
            S, sums = greedy_construct(order)
        elif strategy == 2:
            # Hybrid: structured start + random
            order = list(range(1, N + 1))
            random.shuffle(order)
            spread = [1, N//4, N//2, 3*N//4, N]
            for s in spread:
                if s in order:
                    order.remove(s)
            order = spread + order
            S, sums = greedy_construct(order)
        elif strategy == 3:
            # Middle-biased ordering
            order = list(range(1, N + 1))
            random.shuffle(order)
            order.sort(key=lambda x: -abs(x - N/2) + random.random() * 50)
            S, sums = greedy_construct(order)
        else:
            # Coverage-based: pick elements that block the most
            S = []
            sums = set()
            remaining = set(range(1, N + 1))
            while remaining:
                best_x = None
                best_score = -1
                if len(remaining) > 30:
                    candidates = random.sample(list(remaining), 30)
                else:
                    candidates = list(remaining)
                for x in candidates:
                    if not is_valid_add(x, S, sums):
                        continue
                    new_sums = sums | {2 * x} | {x + s for s in S}
                    blocked = sum(1 for y in remaining if y != x and not is_valid_add(y, S, new_sums))
                    score = blocked * 10 - abs(x - N/2) * 0.1 + random.random() * 5
                    if score > best_score:
                        best_score = score
                        best_x = x
                if best_x is None:
                    break
                S.append(best_x)
                sums.add(2 * best_x)
                for s in S[:-1]:
                    sums.add(best_x + s)
                remaining.remove(best_x)
                if len(S) >= 10 and is_maximal(S, sums):
                    break
        
        if not S:
            continue
        
        # Check if maximal, try to reduce
        if is_maximal(S, sums):
            S_reduced, sums_reduced = local_search_reduce(S, sums, deadline)
            if len(S_reduced) < best_size:
                best_size = len(S_reduced)
                best_set = S_reduced[:]
        else:
            # Try to make maximal by adding
            S_temp = list(S)
            sums_temp = set(sums)
            S_set = set(S_temp)
            for y in range(1, N + 1):
                if y in S_set:
                    continue
                if is_valid_add(y, S_temp, sums_temp):
                    S_temp.append(y)
                    S_set.add(y)
                    sums_temp.add(2 * y)
                    for s in S_temp[:-1]:
                        sums_temp.add(y + s)
            if is_maximal(S_temp, sums_temp):
                S_reduced, sums_reduced = local_search_reduce(S_temp, sums_temp, deadline)
                if len(S_reduced) < best_size:
                    best_size = len(S_reduced)
                    best_set = S_reduced[:]
    
    # Final reduction attempt
    if best_set:
        sums_best = compute_sums(best_set)
        S_final, sums_final = hitting_set_search(best_set, sums_best, deadline)
        if len(S_final) < best_size:
            best_size = len(S_final)
            best_set = S_final[:]
    
    return best_set
