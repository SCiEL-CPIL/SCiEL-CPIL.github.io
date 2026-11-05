import random
import time

def construct():
    N = 180
    vertices = list(range(1, N + 1))
    
    # Precompute forbidden triples
    # 1/a = 1/b + 1/c => 1/a - 1/b = 1/c => (b-a)/(ab) = 1/c => c = ab/(b-a)
    # We need a, b, c to be distinct and in [1, N].
    # Note: For 1/a = 1/b + 1/c to hold with positive integers, we must have b > a and c > a.
    
    # adjacency list: for each vertex v, list of pairs (u, w) such that {v, u, w} is a forbidden triple.
    # If we add v, we cannot have both u and w in the set.
    pairs = {i: [] for i in vertices}
    
    for a in range(1, N + 1):
        for b in range(a + 1, N + 1):
            num = a * b
            den = b - a
            if num % den == 0:
                c = num // den
                if c <= N and c != a and c != b:
                    # Found triple (a, b, c)
                    # a is the smallest.
                    # For a: conflicts with (b, c)
                    pairs[a].append((b, c))
                    # For b: conflicts with (a, c)
                    pairs[b].append((a, c))
                    # For c: conflicts with (a, b)
                    pairs[c].append((a, b))
    
    def is_valid_addition(v, current_set):
        """Check if v can be added to current_set without violating constraints."""
        for (u, w) in pairs[v]:
            if u in current_set and w in current_set:
                return False
        return True

    def greedy_construct(order):
        """Construct a set by greedily adding vertices in the given order."""
        current_set = set()
        for v in order:
            if is_valid_addition(v, current_set):
                current_set.add(v)
        return current_set

    def local_search(initial_set):
        """Improve the initial set using a remove-and-refill strategy."""
        current_set = set(initial_set)
        improved = True
        # Limit local search iterations to prevent getting stuck in long loops
        max_iter = 50
        iter_count = 0
        
        while improved and iter_count < max_iter:
            improved = False
            iter_count += 1
            # Try removing each vertex in the current set and see if we can add more
            removal_candidates = list(current_set)
            random.shuffle(removal_candidates)
            
            for v_to_remove in removal_candidates:
                # Try removing v_to_remove
                temp_set = set(current_set)
                temp_set.discard(v_to_remove)
                
                # Try to greedily add any vertex not in temp_set
                # We should prioritize vertices that were previously blocked by v_to_remove?
                # Or just try all vertices not in temp_set.
                # To be efficient, we can just check all vertices not in temp_set.
                # But N is small (180), so checking all is fine.
                new_set = set(temp_set)
                
                # We need to be careful: adding one vertex might block another.
                # A simple greedy pass over the candidates not in new_set.
                # Let's sort candidates by some heuristic? Or just random order?
                # Random order is safer for diversity.
                candidates = [v for v in vertices if v not in new_set]
                random.shuffle(candidates)
                
                for v_add in candidates:
                    if is_valid_addition(v_add, new_set):
                        new_set.add(v_add)
                
                # If the new set is strictly larger, accept it
                if len(new_set) > len(current_set):
                    current_set = new_set
                    improved = True
                    # Restart the loop to check for further improvements
                    break
        return current_set

    best_set = set()
    best_size = 0
    
    start_time = time.time()
    time_limit = 55.0 # Leave some buffer for overhead
    
    while time.time() - start_time < time_limit:
        # Phase 1: Greedy Construction with random order
        random.shuffle(vertices)
        current_set = greedy_construct(vertices)
        
        # Phase 2: Local Search Improvement
        # Only run local search if we have enough time left
        if time.time() - start_time < time_limit - 5.0:
            improved_set = local_search(current_set)
            if len(improved_set) > best_size:
                best_size = len(improved_set)
                best_set = improved_set.copy()
        else:
            if len(current_set) > best_size:
                best_size = len(current_set)
                best_set = current_set.copy()

    return sorted(best_set)
