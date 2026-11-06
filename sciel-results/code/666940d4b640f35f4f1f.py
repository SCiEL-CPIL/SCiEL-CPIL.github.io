    def generate_sums(vals, limit_val):
        sums = [(0, 0)]
        for i, v in enumerate(vals):
            bit = 1 << i
            new_sums = []
            for s, m in sums:
                ns = s + v
                if ns < limit_val:
                    new_sums.append((ns, m | bit))
            sums.extend(new_sums)
        return sums
