    def add_to(elements, element_set, sums, diffs, x):
        elements.append(x)
        element_set.add(x)
        for a in elements:
            if a != x:
                sums.add(a + x)
                diffs.add(x - a)
                diffs.add(a - x)
        sums.add(2 * x)
        
    def remove_from(elements, element_set, sums, diffs, x):
        elements.remove(x)
        element_set.remove(x)
        for a in elements:
            sums.discard(a + x)
            diffs.discard(x - a)
            diffs.discard(a - x)
        sums.discard(2 * x)
