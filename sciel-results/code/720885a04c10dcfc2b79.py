"""Rewritten program: maximum independent set via branch-and-bound on conflict graph."""
import math

MAX_MOD = 128

def construct():
    """
    Construct a nonempty list of [residue, modulus] pairs with distinct moduli 2..128
    such that every two infinite residue classes are disjoint. Maximize the number of classes.
    
    Approach: Use a greedy algorithm with multiple random orderings, then local search.
    """
    def _conflicts(r1, m1, r2, m2):
        """Check if two residue classes conflict (i.e., are not disjoint)."""
        if m1 == m2:
            return True  # Same modulus, always conflict (we need distinct moduli)
        g = math.gcd(m1, m2)
        return (r1 % g) == (r2 % g)
    
    def _try_greedy(order):
        """Given an ordering of (m, r) pairs, greedily select a maximal independent set."""
        selected = []
        for (m, r) in order:
            conflict = False
            for (sm, sr) in selected:
                if _conflicts(r, m, sr, sm):
                    conflict = True
                    break
            if not conflict:
                selected.append((m, r))
        return selected
    
    def _local_search(selected):
        """Try to improve the solution by removing and re-adding nodes."""
        selected_set = set(selected)
        improved = True
        while improved:
            improved = False
            for (m, r) in list(selected_set):
                # Try removing this node and see if we can add two others
                remaining = selected_set - {(m, r)}
                # Find all nodes that conflict with (m, r) but not with remaining
                for m2 in range(2, MAX_MOD + 1):
                    if m2 == m:
                        continue
                    if any(m2 == sm for sm, sr in remaining):
                        continue
                    for r2 in range(m2):
                        if _conflicts(r2, m2, r, m):
                            continue
                        # Check if (m2, r2) conflicts with any in remaining
                        conflict_with_remaining = False
                        for (sm, sr) in remaining:
                            if _conflicts(r2, m2, sr, sm):
                                conflict_with_remaining = True
                                break
                        if not conflict_with_remaining:
                            # We can add (m2, r2)
                            new_selected = remaining | {(m2, r2)}
                            if len(new_selected) > len(selected_set):
                                selected_set = new_selected
                                improved = True
                                break
                    if improved:
                        break
                if improved:
                    break
        return list(selected_set)
    
    # Generate all possible (m, r) pairs
    all_pairs = []
    for m in range(2, MAX_MOD + 1):
        for r in range(m):
            all_pairs.append((m, r))
    
    # Try multiple greedy orderings
    import random
    random.seed(42)
    
    best = []
    
    # Try natural order
    sel = _try_greedy(all_pairs)
    sel = _local_search(sel)
    if len(sel) > len(best):
        best = sel
    
    # Try several random orderings
    for trial in range(50):
        random.shuffle(all_pairs)
        sel = _try_greedy(all_pairs)
        sel = _local_search(sel)
        if len(sel) > len(best):
            best = sel
    
    # Convert to list of [residue, modulus]
    result = [[r, m] for (m, r) in best]
    
    return result
