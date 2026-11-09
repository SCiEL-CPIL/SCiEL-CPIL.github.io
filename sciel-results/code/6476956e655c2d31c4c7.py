import random
from itertools import combinations

def construct():
    n = 192
    max_elements = 48
    
    # Precompute conflict potential for each number
    # A number x is "conflict-prone" if it divides many possible sums of 3 distinct numbers from 1..n
    # Possible sums range from 1+2+3=6 to (n-2)+(n-1)+n
    min_sum = 1 + 2 + 3
    max_sum = (n - 2) + (n - 1) + n
    
    # We don't need the exact set of sums, just the count of multiples in that range
    # But wait, the sums must be achievable by distinct integers.
    # Approximation: Count multiples of x in [min_sum, max_sum].
    # This is a good proxy for "degree".
    divisible_counts = {}
    for x in range(1, n + 1):
        # Count multiples of x in [min_sum, max_sum]
        # First multiple >= min_sum
        start = ((min_sum + x - 1) // x) * x
        if start > max_sum:
            divisible_counts[x] = 0
        else:
            divisible_counts[x] = (max_sum - start) // x + 1

    def greedy_degree_based(seed=0):
        random.seed(seed)
        selected = []
        selected_set = set()
        
        # sums_mod[a] = set of residues r such that exists b,c in selected\{a} with (b+c)%a == r
        sums_mod = {}
        # triples_sum = set of all sums b+c+d for distinct b,c,d in selected
        triples_sum = set()
        
        for _ in range(max_elements):
            valid_candidates = []
            
            for x in range(1, n + 1):
                if x in selected_set:
                    continue
                
                # Check if x divides any existing triple sum
                # If yes, invalid
                conflict_with_existing = False
                for s in triples_sum:
                    if s % x == 0:
                        conflict_with_existing = True
                        break
                
                if conflict_with_existing:
                    continue
                
                # Check if x creates a conflict with any existing a
                # i.e., exists a in selected, b,c in selected\{a} such that a | (b+c+x)
                # This is equivalent to: exists a in selected such that (-x % a) in sums_mod[a]
                conflict_with_new = False
                for a in selected:
                    target_residue = (-x) % a
                    if target_residue in sums_mod.get(a, set()):
                        conflict_with_new = True
                        break
                
                if conflict_with_new:
                    continue
                
                valid_candidates.append(x)
            
            if not valid_candidates:
                break
            
            # Select candidate based on "degree" heuristic
            # High degree (many divisible sums) first?
            # Let's try sorting by divisible_counts descending.
            # Add some randomness to break ties or select from top k.
            valid_candidates.sort(key=lambda x: divisible_counts[x], reverse=True)
            
            # Pick randomly from the top 5 to introduce variation
            k = min(5, len(valid_candidates))
            x = random.choice(valid_candidates[:k])
            
            selected.append(x)
            selected_set.add(x)
            
            # Update data structures
            # 1. Update sums_mod for existing a
            # For each a in selected (excluding x), add new pairs involving x
            for a in selected:
                if a == x:
                    continue
                if a not in sums_mod:
                    sums_mod[a] = set()
                # Add pairs (x, b) for all b in selected, b != a, b != x
                for b in selected:
                    if b == a or b == x:
                        continue
                    residue = (x + b) % a
                    sums_mod[a].add(residue)
            
            # Note: We also need to initialize sums_mod[x] for future checks?
            # sums_mod[x] will contain residues b+c % x for b,c in selected\{x}.
            # Currently selected has x and others.
            # We need to populate sums_mod[x] with pairs from existing selected (excluding x).
            if x not in sums_mod:
                sums_mod[x] = set()
            others = [y for y in selected if y != x]
            for b, c in combinations(others, 2):
                residue = (b + c) % x
                sums_mod[x].add(residue)
                
            # 2. Update triples_sum
            # Add sums x + b + c for all pairs b,c in selected, b!=x, c!=x, b!=c
            for b, c in combinations(others, 2):
                triples_sum.add(x + b + c)
                
        return selected

    def is_valid(selected):
        """Check if the entire set is valid."""
        for a in selected:
            others = [x for x in selected if x != a]
            for b, c, d in combinations(others, 3):
                if (b + c + d) % a == 0:
                    return False
        return True

    def local_search(selected, seed=0):
        random.seed(seed)
        selected = list(selected)
        selected_set = set(selected)
        
        # Rebuild state for speed
        sums_mod = {}
        triples_sum = set()
        
        # Initialize sums_mod and triples_sum for current selected
        for a in selected:
            sums_mod[a] = set()
        for a in selected:
            others = [x for x in selected if x != a]
            for b, c in combinations(others, 2):
                sums_mod[a].add((b + c) % a)
        for b, c, d in combinations(selected, 3):
            triples_sum.add(b + c + d)

        improved = True
        while improved and len(selected) < max_elements:
            improved = False
            
            # Try adding
            valid_candidates = []
            for x in range(1, n + 1):
                if x in selected_set:
                    continue
                
                # Check conflicts
                conflict = False
                for s in triples_sum:
                    if s % x == 0:
                        conflict = True
                        break
                if conflict:
                    continue
                
                for a in selected:
                    if (-x % a) in sums_mod.get(a, set()):
                        conflict = True
                        break
                if conflict:
                    continue
                
                valid_candidates.append(x)
            
            if valid_candidates:
                valid_candidates.sort(key=lambda x: divisible_counts[x], reverse=True)
                x = random.choice(valid_candidates[:min(5, len(valid_candidates))])
                selected.append(x)
                selected_set.add(x)
                
                # Update state
                for a in selected:
                    if a == x:
                        continue
                    if a not in sums_mod:
                        sums_mod[a] = set()
                    for b in selected:
                        if b == a or b == x:
                            continue
                        sums_mod[a].add((x + b) % a)
                
                if x not in sums_mod:
                    sums_mod[x] = set()
                others = [y for y in selected if y != x]
                for b, c in combinations(others, 2):
                    sums_mod[x].add((b + c) % x)
                
                for b, c in combinations(others, 2):
                    triples_sum.add(x + b + c)
                
                improved = True
                continue
            
            # If no addition, try swap
            # This part is complex to implement with the state efficiently.
            # Given time, let's stick to adding only in the main loop, 
            # and rely on multiple randomized restarts.
            # The current program has a swap mechanism. 
            # Implementing swap with state rollback is tricky.
            # Let's skip complex swap and rely on restarts.
            break
            
        return selected

    best = []
    best_size = 0
    
    for trial in range(50):
        s = greedy_degree_based(seed=trial)
        # Simple local search: try to add any valid number (random order)
        # The greedy already does this.
        # Let's just verify and keep the best.
        if is_valid(s):
            if len(s) > best_size:
                best = s
                best_size = len(s)
                if best_size >= max_elements:
                    break

    return best
