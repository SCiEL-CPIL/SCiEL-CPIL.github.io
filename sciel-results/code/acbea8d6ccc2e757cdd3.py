    for s in range(2, 1008):
        # ...
        e = s
        # Try to extend
        while e < 1007:
            # ...
            if inv[new_e] < s or inv[new_e] > new_e:
                e = new_e
            else:
                break
