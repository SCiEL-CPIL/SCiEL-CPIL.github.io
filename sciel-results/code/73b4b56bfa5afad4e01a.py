def construct():
    import random

    def instance(size, seed, limit=10000):
        rng = random.Random(seed)
        values = set()
        while len(values) < size:
            x = rng.randint(-limit, limit)
            if x:
                values.add(x)
        return sorted(values)

    def greedy_with_local_search(elements, seed):
        n = len(elements)
        if n == 0:
            return []
        
        rng = random.Random(seed)
        
        # Greedy initialization: sort by |value| descending, add if possible
        sorted_elems = sorted(elements, key=lambda x: -abs(x))
        selected = set()
        for elem in sorted_elems:
            can_add = True
            for a in selected:
                if (elem - a) in selected:
                    can_add = False
                    break
            if can_add:
                for b in selected:
                    if (b - elem) in selected:
                        can_add = False
                        break
            if can_add:
                if (2 * elem) in selected:
                    can_add = False
            if can_add:
                selected.add(elem)
        
        # Local search: try to add unselected elements
        unselected = [e for e in elements if e not in selected]
        
        # Try adding each unselected element
        changed = True
        while changed:
            changed = False
            for elem in unselected[:]:
                can_add = True
                for a in selected:
                    if (elem - a) in selected:
                        can_add = False
                        break
                if can_add:
                    for b in selected:
                        if (b - elem) in selected:
                            can_add = False
                            break
                if can_add:
                    if (2 * elem) in selected:
                        can_add = False
                if can_add:
                    selected.add(elem)
                    unselected.remove(elem)
                    changed = True
        
        # Try swaps: remove one selected, try to add multiple unselected
        for _ in range(50):
            improved = False
            for sel_elem in list(selected):
                trial_selected = selected - {sel_elem}
                added_any = False
                for elem in unselected[:]:
                    can_add = True
                    for a in trial_selected:
                        if (elem - a) in trial_selected:
                            can_add = False
                            break
                    if can_add:
                        for b in trial_selected:
                            if (b - elem) in trial_selected:
                                can_add = False
                                break
                    if can_add:
                        if (2 * elem) in trial_selected:
                            can_add = False
                    if can_add:
                        trial_selected.add(elem)
                        unselected.remove(elem)
                        added_any = True
                if len(trial_selected) > len(selected):
                    selected = trial_selected
                    improved = True
                    break
            if not improved:
                break
        
        return list(selected)

    results = []
    for size, seed in ((60, 801), (120, 802), (240, 803)):
        elems = instance(size, seed)
        subset = greedy_with_local_search(elems, seed)
        results.append(subset)

    return results
