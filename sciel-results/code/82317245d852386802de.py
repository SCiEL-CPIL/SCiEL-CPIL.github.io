import random
import time
from math import gcd

def construct():
    start_time = time.time()
    deadline = start_time + 55.0
    
    N = 128
    # Precompute GCD table
    gcd_table = [[0]*(N+1) for _ in range(N+1)]
    for i in range(1, N+1):
        for j in range(i, N+1):
            g = gcd(i, j)
            gcd_table[i][j] = g
            gcd_table[j][i] = g
            
    # Build conflict graph: edge between a,b if they participate in a bad triple together
    # Two elements a,b conflict if there exists c such that (a,b,c) is a bad triple
    # We'll build adjacency for the conflict graph
    conflict = [set() for _ in range(N+1)]
    
    for a in range(1, N+1):
        for b in range(a+1, N+1):
            g_ab = gcd_table[a][b]
            for c in range(b+1, N+1):
                g_bc = gcd_table[b][c]
                if g_ab == g_bc:
                    g_ca = gcd_table[c][a]
                    if g_ab == g_ca:
                        # (a,b,c) is bad
                        conflict[a].add(b)
                        conflict[a].add(c)
                        conflict[b].add(a)
                        conflict[b].add(c)
                        conflict[c].add(a)
                        conflict[c].add(b)
    
    def is_valid_set(S):
        """Check if set S has no bad triples."""
        S_set = set(S)
        for a in S:
            for b in S:
                if a >= b:
                    continue
                g_ab = gcd_table[a][b]
                for c in S:
                    if c <= b:
                        continue
                    if gcd_table[a][c] == g_ab and gcd_table[b][c] == g_ab:
                        return False
        return True
    
    def count_bad_triples_in(S):
        """Count number of bad triples within set S."""
        S_set = set(S)
        count = 0
        for a in S:
            for b in S:
                if a >= b:
                    continue
                g_ab = gcd_table[a][b]
                for c in S:
                    if c <= b:
                        continue
                    if gcd_table[a][c] == g_ab and gcd_table[b][c] == g_ab:
                        count += 1
        return count
    
    def simulated_annealing(initial_set, max_iterations=50000):
        """Run simulated annealing starting from initial_set."""
        current = set(initial_set)
        best = set(initial_set)
        best_size = len(initial_set)
        
        T = 2.0
        cooling = 0.99995
        
        for it in range(max_iterations):
            if time.time() > deadline:
                break
            
            # Try a random move: add, remove, or swap
            move_type = random.random()
            
            if move_type < 0.4:
                # Add a random element not in current
                candidates = [x for x in range(1, N+1) if x not in current]
                if not candidates:
                    continue
                x = random.choice(candidates)
                new_set = current | {x}
                delta = 1  # size increases by 1
            elif move_type < 0.7:
                # Remove a random element from current
                if not current:
                    continue
                x = random.choice(list(current))
                new_set = current - {x}
                delta = -1  # size decreases by 1
            else:
                # Swap: remove one, add one
                if not current:
                    continue
                x_remove = random.choice(list(current))
                candidates = [x for x in range(1, N+1) if x not in current]
                if not candidates:
                    continue
                x_add = random.choice(candidates)
                new_set = (current - {x_remove}) | {x_add}
                delta = 0  # size unchanged
            
            # Evaluate the move: check if new_set is valid
            if is_valid_set(new_set):
                # Accept if valid
                current = new_set
                if len(current) > best_size:
                    best_size = len(current)
                    best = set(current)
            else:
                # Invalid: accept with probability based on how many bad triples
                # For simplicity, reject invalid moves (since we want max valid set)
                # But we could use a penalty-based approach
                bad_count = count_bad_triples_in(new_set)
                if bad_count > 0:
                    # Reject
                    continue
                else:
                    # Should be valid
                    current = new_set
                    if len(current) > best_size:
                        best_size = len(current)
                        best = set(current)
            
            T *= cooling
            
        return best
    
    # Multi-start: generate diverse initial solutions
    best_overall = set()
    best_size_overall = 0
    
    # Start 1: Greedy construction
    greedy_set = set()
    remaining = list(range(1, N+1))
    random.shuffle(remaining)
    for x in remaining:
        test_set = greedy_set | {x}
        if is_valid_set(test_set):
            greedy_set = test_set
    
    if len(greedy_set) > best_size_overall:
        best_size_overall = len(greedy_set)
        best_overall = set(greedy_set)
    
    # Start 2: All elements, then remove conflicts greedily
    all_set = set(range(1, N+1))
    # Remove elements in order of highest conflict degree first
    degrees = {x: len(conflict[x]) for x in range(1, N+1)}
    order = sorted(range(1, N+1), key=lambda x: -degrees[x])
    removal_set = set(all_set)
    for x in order:
        if x in removal_set:
            # Check if x conflicts with any remaining element
            has_conflict = False
            for y in removal_set:
                if y != x and y in conflict[x]:
                    has_conflict = True
                    break
            if has_conflict:
                removal_set.discard(x)
    
    if len(removal_set) > best_size_overall:
        best_size_overall = len(removal_set)
        best_overall = set(removal_set)
    
    # Run simulated annealing from multiple starting points
    starts = [greedy_set, removal_set]
    
    # Add some random starts
    for _ in range(3):
        if time.time() > deadline:
            break
        rand_set = set(random.sample(range(1, N+1), random.randint(20, 50)))
        # Clean up: remove elements that cause bad triples
        cleaned = set()
        for x in sorted(rand_set):
            test = cleaned | {x}
            if is_valid_set(test):
                cleaned = test
        starts.append(cleaned)
    
    for start_set in starts:
        if time.time() > deadline:
            break
        result = simulated_annealing(start_set, max_iterations=30000)
        if len(result) > best_size_overall:
            best_size_overall = len(result)
            best_overall = set(result)
    
    # Local search: try to add back elements
    current = set(best_overall)
    removed = set(range(1, N+1)) - current
    rem_list = list(removed)
    random.shuffle(rem_list)
    
    for x in rem_list:
        if time.time() > deadline:
            break
        test = current | {x}
        if is_valid_set(test):
            current = test
    
    # 1-1 swap local search
    improved = True
    while improved:
        improved = False
        if time.time() > deadline:
            break
        current_list = list(current)
        removed_list = [x for x in range(1, N+1) if x not in current]
        random.shuffle(removed_list)
        
        for x_remove in current_list:
            if time.time() > deadline:
                break
            for x_add in removed_list:
                if x_add in current:
                    continue
                test = (current - {x_remove}) | {x_add}
                if is_valid_set(test):
                    # Check if this swap helps (it shouldn't since size is same, but maybe enables future adds)
                    # For now, accept swaps that maintain validity
                    current = test
                    improved = True
                    break
            if improved:
                break
    
    if len(current) > best_size_overall:
        best_size_overall = len(current)
        best_overall = set(current)
    
    return list(best_overall)
