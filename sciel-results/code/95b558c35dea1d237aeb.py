import random
from collections import defaultdict

def construct():
    # Enumerate all forbidden triples (a,b,c) distinct, in [1,180], 1/a=1/b+1/c
    triples = []
    for b in range(1, 181):
        for c in range(b+1, 181):
            s = b + c
            prod = b * c
            if prod % s == 0:
                a = prod // s
                if 1 <= a <= 180 and a != b and a != c:
                    triples.append(tuple(sorted((a, b, c))))
    
    # Remove duplicates
    triples = list(set(triples))
    
    # Pre-compute for each element, the list of triples containing it
    elem_to_triples = defaultdict(list)
    for i, t in enumerate(triples):
        for e in t:
            elem_to_triples[e].append(i)
    
    def greedy_hitting_set(triple_list, seed=42):
        random.seed(seed)
        remaining = set(range(1, 181))
        active = set(range(len(triple_list)))
        removed = set()
        
        while active:
            # Count occurrences of each element in active triples
            cnt = defaultdict(int)
            for ti in active:
                for e in triple_list[ti]:
                    cnt[e] += 1
            if not cnt:
                break
            # Pick element with max count (tie-break: smallest)
            worst = min(cnt.keys(), key=lambda e: (-cnt[e], e))
            removed.add(worst)
            remaining.discard(worst)
            # Remove all active triples containing worst
            to_remove = set()
            for ti in elem_to_triples[worst]:
                if ti in active:
                    to_remove.add(ti)
            active -= to_remove
        
        return remaining, removed
    
    def relax(remaining, removed, triple_list, elem_to_triples, max_passes=50):
        """Try to re-add elements from removed to remaining if possible."""
        remaining = set(remaining)
        removed = set(removed)
        
        for _ in range(max_passes):
            improved = False
            # Try to re-add each element in removed (in random order)
            for e in list(removed):
                # Check if we can add e to remaining
                # A triple T containing e is a conflict if T\{e} ⊆ remaining
                can_add = True
                for ti in elem_to_triples[e]:
                    t = triple_list[ti]
                    others = [x for x in t if x != e]
                    if all(x in remaining for x in others):
                        can_add = False
                        break
                if can_add:
                    remaining.add(e)
                    removed.discard(e)
                    improved = True
            if not improved:
                break
        
        return remaining, removed
    
    # Try multiple seeds and strategies
    best_set = set(range(1, 181))
    best_size = 0
    
    for seed in [42, 123, 777, 999, 2024, 555, 1337, 4242]:
        remaining, removed = greedy_hitting_set(triples, seed=seed)
        remaining, removed = relax(remaining, removed, triples, elem_to_triples)
        
        if len(remaining) > best_size:
            best_size = len(remaining)
            best_set = set(remaining)
    
    # Also try a different greedy: prefer removing smaller numbers
    def greedy_hitting_set_keep_large(triple_list, seed=42):
        random.seed(seed)
        remaining = set(range(1, 181))
        active = set(range(len(triple_list)))
        removed = set()
        
        while active:
            cnt = defaultdict(int)
            for ti in active:
                for e in triple_list[ti]:
                    cnt[e] += 1
            if not cnt:
                break
            # Among max count, remove the smallest (to keep larger numbers)
            worst = min(cnt.keys(), key=lambda e: (-cnt[e], e))
            removed.add(worst)
            remaining.discard(worst)
            to_remove = set()
            for ti in elem_to_triples[worst]:
                if ti in active:
                    to_remove.add(ti)
            active -= to_remove
        
        return remaining, removed
    
    for seed in [42, 123, 777, 999, 2024, 555, 1337, 4242]:
        remaining, removed = greedy_hitting_set_keep_large(triples, seed=seed)
        remaining, removed = relax(remaining, removed, triples, elem_to_triples)
        
        if len(remaining) > best_size:
            best_size = len(remaining)
            best_set = set(remaining)
    
    # Additional local search: try swapping elements
    def local_search_improve(remaining, triple_list, elem_to_triples, iterations=200):
        remaining = set(remaining)
        removed = set(range(1, 181)) - remaining
        
        for _ in range(iterations):
            improved = False
            # Try to add a random removed element
            if removed:
                e = random.choice(list(removed))
                can_add = True
                for ti in elem_to_triples[e]:
                    t = triple_list[ti]
                    others = [x for x in t if x != e]
                    if all(x in remaining for x in others):
                        can_add = False
                        break
                if can_add:
                    remaining.add(e)
                    removed.discard(e)
                    improved = True
            
            # Try to remove a random remaining element that is "expensive"
            if remaining and not improved:
                # Find element in most active triples
                cnt = defaultdict(int)
                for ti in range(len(triple_list)):
                    t = triple_list[ti]
                    if all(x in remaining for x in t):
                        for x in t:
                            cnt[x] += 1
                if cnt:
                    worst = max(cnt.keys(), key=lambda e: (cnt[e], -e))
                    remaining.discard(worst)
                    removed.add(worst)
                    improved = True
            
            if not improved:
                break
        
        return remaining
    
    for seed in [42, 123, 777, 999, 2024, 555, 1337, 4242]:
        random.seed(seed)
        current = set(best_set)
        improved_set = local_search_improve(current, triples, elem_to_triples, iterations=100)
        if len(improved_set) > best_size:
            best_size = len(improved_set)
            best_set = improved_set
    
    return sorted(best_set)
