        # Check Sidon property
        sums = {}
        for i, a in enumerate(selected):
            for j in range(i, len(selected)):
                b = selected[j]
                s = a + b
                if s in sums:
                    return False, False
                sums[s] = (a, b)
        
        # Check maximality: no missing element can be added
        selected_set = set(selected)
        for x in range(1, N + 1):
            if x in selected_set:
                continue
            # Try adding x
            new_sums = [x + a for a in selected] + [2 * x]
            if len(set(new_sums)) != len(new_sums):
                continue
            if sums.intersection(new_sums):  # ERROR HERE: sums is a dict
                continue
            # x can be added, so not maximal
            return True, False
        
        return True, True
