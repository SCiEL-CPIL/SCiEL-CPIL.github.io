import itertools
import time

def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # 80s safety margin out of 90s
    
    n = 12
    m = 40
    
    # Generate all subsets of {0..11} as frozensets for fast subset checks
    # There are 2^12 = 4096 subsets
    all_subsets = []
    for mask in range(1 << n):
        s = frozenset(b for b in range(n) if (mask >> b) & 1)
        all_subsets.append(s)
    
    # Precompute subset relationships: A is a strict subset of B
    # For each pair, check if A < B (strict subset)
    # We'll build a graph where edge (i, j) means all_subsets[i] is strict subset of all_subsets[j]
    
    # For efficiency, group by cardinality
    by_size = {}
    for i, s in enumerate(all_subsets):
        sz = len(s)
        if sz not in by_size:
            by_size[sz] = []
        by_size[sz].append(i)
    
    # For each subset, find all strict supersets
    # strict_supersets[i] = set of j such that all_subsets[i] is strict subset of all_subsets[j]
    # This is expensive to precompute fully, so let's be smarter.
    
    # Actually, let's think about this differently.
    # The objective is to maximize the number of comparable pairs among 40 selected sets.
    # A pair (A, B) is comparable if A is a strict subset of B or B is a strict subset of A.
    
    # Key insight from the knowledge: working within B_6 (subsets of a 6-element set) 
    # allows for denser comparability. The 40 smallest sets of B_6 gave a good score.
    
    # Let's try a greedy approach on the full 12-element Boolean lattice,
    # but initialize with the B_6-based family and then try to improve.
    
    # First, let's create the B_6-based family as initial solution
    # B_6: subsets of {0,1,2,3,4,5}
    b6_sets = []
    for r in range(7):
        for combo in itertools.combinations(range(6), r):
            b6_sets.append(frozenset(combo))
    
    # Sort by size
    b6_sets.sort(key=lambda s: (len(s), s))
    
    # Take 40 smallest
    initial_selection = list(b6_sets[:40])
    
    # Convert to indices in all_subsets
    # Build a map from frozenset to index
    set_to_idx = {}
    for i, s in enumerate(all_subsets):
        set_to_idx[s] = i
    
    selected = [set_to_idx[s] for s in initial_selection]
    selected_set = set(selected)
    
    def count_comparable(sel_set):
        """Count comparable pairs in a set of subset indices."""
        count = 0
        sel_list = list(sel_set)
        for i in range(len(sel_list)):
            for j in range(i + 1, len(sel_list)):
                a = all_subsets[sel_list[i]]
                b = all_subsets[sel_list[j]]
                if a < b or b < a:
                    count += 1
        return count
    
    current_score = count_comparable(selected_set)
    
    # Now try greedy improvement: try swapping each selected set with an unselected one
    # if it improves the score
    
    # Precompute for each subset, which other subsets it's comparable with
    # comparable_with[i] = set of j such that all_subsets[i] and all_subsets[j] are comparable
    # This is expensive: 4096 * 4096 / 2 = ~8 million checks
    # Each check is a frozenset comparison which is O(n) worst case
    
    # Let's precompute this
    print(f"Precomputing comparability...")
    comparable_with = [set() for _ in range(1 << n)]
    
    for i in range(1 << n):
        a = all_subsets[i]
        for j in range(i + 1, 1 << n):
            b = all_subsets[j]
            if a < b:
                comparable_with[i].add(j)
                comparable_with[j].add(i)
            elif b < a:
                comparable_with[i].add(j)
                comparable_with[j].add(i)
    
    print(f"Precomputation done. Time: {time.time() - start_time:.1f}s")
    
    # Now greedy local search
    # For each selected element, try swapping with an unselected element
    # Gain from swapping out x and in y:
    # - lose comparabilities between x and other selected (except y if y was comparable)
    # - gain comparabilities between y and other selected (except x if x was comparable)
    
    unselected = set(range(1 << n)) - selected_set
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        best_gain = 0
        best_swap = None
        
        for x in list(selected_set):
            # For each unselected y, compute gain
            # This is |selected| * |unselected| which is 40 * 4056 = 162240 per full pass
            # Too slow for many iterations. Let's be smarter.
            
            # Actually, let's just do a few iterations of this
            break
        
        # Simpler approach: random restarts with greedy construction
        # Let's try a different strategy: greedy construction from scratch
        
        # Greedy: start with empty set, add one set at a time that maximizes new comparable pairs
        # But we need exactly 40 sets.
        
        # Let's do: for each of the 40 slots, pick the set that maximizes total comparable pairs
        # with already selected sets.
        
        # This is O(m * 2^n * m) = 40 * 4096 * 40 = 6.5 million operations
        # Each operation is a set lookup, so should be fast.
        
        if time.time() < deadline - 10:
            # Greedy construction
            greedy_selected = set()
            for step in range(m):
                best_set = None
                best_score = -1
                for s_idx in range(1 << n):
                    if s_idx in greedy_selected:
                        continue
                    # Count comparable with already selected
                    score = len(comparable_with[s_idx] & greedy_selected)
                    if score > best_score:
                        best_score = score
                        best_set = s_idx
                greedy_selected.add(best_set)
            
            greedy_score = count_comparable(greedy_selected)
            if greedy_score > current_score:
                selected_set = greedy_selected
                current_score = greedy_score
                print(f"Greedy improved to {current_score}")
        
        # Try local search on current best
        if time.time() < deadline - 5:
            for _ in range(100):  # 100 local search iterations
                if time.time() > deadline - 2:
                    break
                # Pick a random selected element to swap out
                x = list(selected_set)[0]  # Just use first for simplicity in this version
                # Actually, let's do proper random swaps
                import random
                random.seed(42)
                x = random.choice(list(selected_set))
                y = random.choice(list(unselected))
                
                # Compute gain
                # Lose: |comparable_with[x] & (selected_set - {x})|
                # Gain: |comparable_with[y] & (selected_set - {x})|
                # Note: if x and y are comparable, we lose that pair and gain it back, so net 0 for that pair
                # Actually: 
                # Old pairs involving x: |comparable_with[x] & (selected_set - {x})|
                # New pairs involving y: |comparable_with[y] & (selected_set - {x})|
                # Pairs not involving x or y are unchanged.
                # So gain = |comparable_with[y] & (selected_set - {x})| - |comparable_with[x] & (selected_set - {x})|
                
                sel_minus_x = selected_set - {x}
                old_pairs = len(comparable_with[x] & sel_minus_x)
                new_pairs = len(comparable_with[y] & sel_minus_x)
                gain = new_pairs - old_pairs
                
                if gain > 0:
                    selected_set.remove(x)
                    selected_set.add(y)
                    unselected.remove(y)
                    unselected.add(x)
                    current_score += gain
                    improved = True
                    if time.time() > deadline - 2:
                        break
    
    # Convert back to list of lists
    result = []
    for s_idx in selected_set:
        s = all_subsets[s_idx]
        result.append(sorted(list(s)))
    
    # Ensure exactly 40 distinct sets
    assert len(result) == 40
    assert len(set(frozenset(s) for s in result)) == 40
    
    return result
