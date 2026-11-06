import time
import itertools
import random
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = 115.0  # Safety margin before 120s limit

    points = list(range(16))
    all_triples = list(itertools.combinations(points, 3))
    
    # Precompute pairs for each triple
    triple_pairs = {t: set(itertools.combinations(t, 2)) for t in all_triples}
    
    # Map pair -> triples containing it
    pair_to_triples = defaultdict(list)
    for t in all_triples:
        for p in triple_pairs[t]:
            pair_to_triples[p].append(t)
    
    def get_alpha(triples):
        """
        Compute the independence number of the hypergraph defined by `triples`.
        An independent set is a set of points containing no entire triple.
        """
        if not triples:
            return 16
            
        # Convert to a set of frozensets for fast lookup? 
        # Or just use the list.
        # To check if a subset S is independent:
        #   No triple t is such that set(t) <= set(S).
        
        # We can iterate subsets of size k from 16 down to 1.
        # But we want the MAX k such that an independent set of size k exists.
        # So we check k=16, 15, ..., 1.
        # If we find one, return k.
        
        # Optimization:
        # If alpha < 5 is our goal, we can stop early if we prove no set of size 5 is independent.
        # However, we need the exact value.
        
        # Let's try to find the max independent set using a heuristic/branch and bound if needed,
        # but for 16 points, brute force checking subsets of size k is feasible for small k.
        # C(16, 10) = 8008. C(16, 8) = 12870. C(16, 6) = 8008.
        # Total subsets = 2^16 = 65536.
        # We can just iterate ALL subsets and find the largest independent one.
        # 65536 subsets * (check if independent)
        # Check: for each subset mask, check if any triple is contained in it.
        # Represent triples as bitmasks.
        
        triple_masks = []
        for t in triples:
            mask = 0
            for p in t:
                mask |= (1 << p)
            triple_masks.append(mask)
            
        max_alpha = 0
        # Iterate all non-empty subsets
        # Actually, we can iterate from largest size down and return the first found.
        # But iterating all masks 1..(2^16-1) and checking independence is fast enough in Python if optimized?
        # 65k iterations.
        
        # Let's precompute which masks are "blocked" by a triple?
        # A subset S is independent if it does NOT contain any triple.
        
        # Alternative: Iterate k from 16 down to 1.
        # For each k, check if there exists an independent set of size k.
        # Stop at the first k found.
        
        # To check existence of independent set of size k:
        # Iterate all C(16, k) subsets.
        # For each subset, check if it contains any triple.
        
        # Let's use a bitset approach for the check.
        # For a given subset mask S, it is independent if for all t in triple_masks:
        #   (S & t_mask) != t_mask  (i.e., t is not a subset of S)
        
        # This check is O(|triples|).
        
        # Let's try k=16 down to 1.
        # If k is large, C(16, k) is small.
        # If k is small, C(16, k) is small.
        # Max C(16, k) is at k=8, which is 12870.
        # So worst case per k is ~13k subsets * 40 triples = 500k ops.
        # Total for all k is sum C(16,k) = 65536 subsets * 40 ops = 2.6M ops.
        # This is very fast in Python (< 0.1s).
        
        for k in range(16, 0, -1):
            for subset in itertools.combinations(points, k):
                mask = 0
                for p in subset:
                    mask |= (1 << p)
                
                is_independent = True
                for t_mask in triple_masks:
                    if (mask & t_mask) == t_mask:
                        is_independent = False
                        break
                
                if is_independent:
                    return k
        return 1

    # Strategy:
    # 1. Generate a random linear triple system (packing).
    # 2. Use a local search to minimize alpha.
    #    Move: Swap a triple with another valid triple.
    #    To speed up, we can also try adding/removing triples if the system isn't maximal?
    #    Actually, maximizing the number of triples usually helps minimize alpha, but not always.
    #    Let's stick to fixed size swaps first, or allow size changes?
    #    The problem doesn't specify a fixed number of triples.
    #    "construct() -> list of triples".
    #    So we can add or remove triples.
    
    # Better Strategy:
    # Start with a random maximal packing (greedy).
    # Then perform local search:
    #   - Try to replace a triple with another.
    #   - Try to add a triple if possible.
    #   - Try to remove a triple if it helps (unlikely, but possible if it allows adding many others? No, adding is always good for density).
    #   Actually, removing a triple frees up pairs, allowing potentially better triples to be added.
    #   So "Remove one, Add one or more" is a valid move.
    
    # Let's implement a simple hill-climbing / simulated annealing.
    
    best_alpha = 16
    best_system = []
    
    # Initialize with a few random greedy packings
    for _ in range(10):
        random.shuffle(all_triples)
        current = []
        used_pairs = set()
        for t in all_triples:
            if all(p not in used_pairs for p in triple_pairs[t]):
                current.append(t)
                for p in triple_pairs[t]:
                    used_pairs.add(p)
        
        alpha = get_alpha(current)
        if alpha < best_alpha:
            best_alpha = alpha
            best_system = current[:]
            
        if best_alpha <= 3:
            break
    
    current_system = best_system[:]
    current_alpha = best_alpha
    
    # Local Search
    # We will use a simple random restart local search with a time limit.
    
    def can_add(triple, used_pairs):
        return all(p not in used_pairs for p in triple_pairs[triple])
    
    while time.time() - start_time < deadline:
        # Random restart occasionally?
        # Or just keep searching the current basin.
        
        # Generate a move
        # 1. Pick a triple to remove
        # 2. Pick a new triple to add (must be valid after removal)
        # Or just pick a new triple to add to the current system?
        # If we just add, the system grows.
        # If we swap, the size stays constant.
        
        # Let's try both:
        # Move 1: Add a random valid triple.
        # Move 2: Swap a random triple with a random valid triple.
        
        move_type = random.choice(['add', 'swap'])
        
        if move_type == 'add':
            # Find all valid triples to add
            used_pairs = set()
            for t in current_system:
                used_pairs.update(triple_pairs[t])
            
            candidates = []
            for t in all_triples:
                if can_add(t, used_pairs):
                    candidates.append(t)
            
            if candidates:
                new_t = random.choice(candidates)
                new_system = current_system + [new_t]
                new_alpha = get_alpha(new_system)
                
                if new_alpha < current_alpha:
                    current_system = new_system
                    current_alpha = new_alpha
                    if current_alpha < best_alpha:
                        best_alpha = current_alpha
                        best_system = current_system[:]
                elif new_alpha == current_alpha:
                    # Accept to explore
                    current_system = new_system
                    # Don't update best_alpha unless it's better
                    # But we should keep track of the best system seen
                    # Actually, if alpha is same, the system might be better positioned for future moves.
                    # Let's accept it.
        
        elif move_type == 'swap':
            if not current_system:
                continue
                
            idx = random.randint(0, len(current_system) - 1)
            t_remove = current_system[idx]
            
            # Remove it temporarily to find candidates
            used_pairs = set()
            for i, t in enumerate(current_system):
                if i != idx:
                    used_pairs.update(triple_pairs[t])
            
            candidates = []
            for t in all_triples:
                if can_add(t, used_pairs):
                    candidates.append(t)
            
            if candidates:
                new_t = random.choice(candidates)
                if new_t == t_remove:
                    continue
                    
                new_system = current_system[:]
                new_system[idx] = new_t
                new_alpha = get_alpha(new_system)
                
                if new_alpha < current_alpha:
                    current_system = new_system
                    current_alpha = new_alpha
                    if current_alpha < best_alpha:
                        best_alpha = current_alpha
                        best_system = current_system[:]
                elif new_alpha == current_alpha:
                    # Accept with some probability? Or always?
                    # Always accepting equal moves leads to random walk.
                    # Let's accept 50% of the time to escape local minima.
                    if random.random() < 0.5:
                        current_system = new_system
        
        if best_alpha <= 3:
            break
    
    return [list(t) for t in best_system]

if __name__ == "__main__":
    res = construct()
    print(res)
