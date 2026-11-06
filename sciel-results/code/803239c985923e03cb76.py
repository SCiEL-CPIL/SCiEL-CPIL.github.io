import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 85s deadline, 5s safety margin
    
    all_sets = [frozenset(c) for c in combinations(range(1, 17), 4)]
    n_total = len(all_sets)  # 1820
    
    best_family = []
    best_size = 0
    
    def can_add(s, family):
        """Check if s can be added to family without creating a 3-sunflower."""
        n = len(family)
        if n < 2:
            return True
        # Group existing sets by their intersection with s
        groups = {}
        for a in family:
            inter = s & a
            groups.setdefault(inter, []).append(a)
        
        # For each group with intersection I, check if any pair in the group
        # has intersection equal to I (which would form a sunflower with s)
        for I, members in groups.items():
            if len(members) < 2:
                continue
            m = len(members)
            for i in range(m):
                a = members[i]
                for j in range(i + 1, m):
                    b = members[j]
                    if a & b == I:
                        return False
        return True
    
    def greedy_pass(order):
        """Greedy: add sets in given order if no sunflower is formed."""
        family = []
        for s in order:
            if can_add(s, family):
                family.append(s)
        return family
    
    def improve(family):
        """Try to add more sets to family via iterative improvement."""
        family_set = set(family)
        improved = True
        while improved and time.time() < deadline - 10.0:
            improved = False
            remaining = [s for s in all_sets if s not in family_set]
            random.shuffle(remaining)
            for s in remaining:
                if can_add(s, family):
                    family.append(s)
                    family_set.add(s)
                    improved = True
        return family
    
    # Multiple random greedy passes with improvement
    passes = 0
    while time.time() < deadline - 15.0:
        random.shuffle(all_sets)
        family = greedy_pass(all_sets)
        family = improve(family)
        passes += 1
        
        if len(family) > best_size:
            best_size = len(family)
            best_family = family[:]
        
        # Early stop if we've found a large family
        if best_size >= 200:
            break
    
    return [sorted(s) for s in best_family]
