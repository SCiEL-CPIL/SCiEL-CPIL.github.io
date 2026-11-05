    def greedy_sidon_size(subset):
        elems = sorted(subset)
        current = []
        sums = set()
        for e in elems:
            can_add = True
            for c in current:
                if e + c in sums:
                    can_add = False
                    break
            if can_add:
                if e + e in sums:
                    can_add = False
            if can_add:
                for c in current:
                    sums.add(e + c)
                sums.add(e + e)
                current.append(e)
        return len(current)
    