<<<<<<< SEARCH
    def greedy_init(elements, order_seed=0):
        """Greedy initialization with a specific ordering."""
        n = len(elements)
        rng = Random(order_seed)
        
        # Try different orderings: by abs value, random, by value
        orderings = [
            sorted(range(n), key=lambda i: -abs(elements[i])),
            sorted(range(n), key=lambda i: elements[i]),
            sorted(range(n), key=lambda i: -elements[i]),
            list(range(n)),
        ]
        rng.shuffle(orderings)
        
        best_subset = []
        best_size = 0
        
        for order in orderings:
            included = set()
            for idx in order:
                c_val = elements[idx]
                ok = True
                # Check 1: is there a, b in included with a+b = c_val?
                for a_idx in included:
                    a_val = elements[a_idx]
                    b_val = c_val - a_val
                    if b_val in {elements[i] for i in included}:
                        ok = False
                        break
                if not ok:
                    continue
                # Check 2: is there a in included with a + c_val in included?
                for a_idx in included:
                    a_val = elements[a_idx]
                    if a_val + c_val in {elements[i] for i in included}:
                        ok = False
                        break
                if not ok:
                    continue
                # Check 3: is 2*c_val in included?
                if 2 * c_val in {elements[i] for i in included}:
                    ok = False
                    continue
                if ok:
                    included.add(idx)
            
            if len(included) > best_size:
                best_size = len(included)
                best_subset = list(included)
        
        return [elements[i] for i in best_subset]
=======
    def greedy_init(elements, order_seed=0):
        """Greedy initialization with a specific ordering."""
        n = len(elements)
        rng = Random(order_seed)
        
        # Try different orderings: by abs value, random, by value
        orderings = [
            sorted(range(n), key=lambda i: -abs(elements[i])),
            sorted(range(n), key=lambda i: elements[i]),
            sorted(range(n), key=lambda i: -elements[i]),
            list(range(n)),
            sorted(range(n), key=lambda i: elements[i] % 7),
        ]
        rng.shuffle(orderings)
        
        best_subset = []
        best_size = 0
        
        for order in orderings:
            included = set()
            included_vals = set()
            for idx in order:
                c_val = elements[idx]
                ok = True
                # Check 1: is there a, b in included with a+b = c_val?
                for a_val in included_vals:
                    b_val = c_val - a_val
                    if b_val in included_vals:
                        ok = False
                        break
                if not ok:
                    continue
                # Check 2: is there a in included with a + c_val in included?
                for a_val in included_vals:
                    if a_val + c_val in included_vals:
                        ok = False
                        break
                if not ok:
                    continue
                # Check 3: is 2*c_val in included?
                if 2 * c_val in included_vals:
                    ok = False
                    continue
                if ok:
                    included.add(idx)
                    included_vals.add(c_val)
            
            if len(included) > best_size:
                best_size = len(included)
                best_subset = list(included)
        
        return [elements[i] for i in best_subset]
>>>>>>> REPLACE

