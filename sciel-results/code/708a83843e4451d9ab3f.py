    def construct():
        import time
        from fractions import Fraction
        start = time.time()
        deadline = start + 50.0
        
        def search(k):
            # DFS for k terms
            def dfs(rem, last, depth, chosen):
                if time.time() > deadline: return None
                if depth == k:
                    if rem == 0: return list(chosen)
                    return None
                if rem <= 0: return None
                # Prune: max possible sum
                # terms left = k - depth
                # smallest denoms = last+1 .. last + (k-depth)
                max_sum = sum(Fraction(1, last+i) for i in range(1, k-depth+1))
                if rem > max_sum: return None
                
                # min den
                # 1/d <= rem => d >= 1/rem
                inv_rem = Fraction(1,1) / rem
                d_min = (inv_rem.numerator + inv_rem.denominator - 1) // inv_rem.denominator
                if d_min < last + 1: d_min = last + 1
                
                for d in range(d_min, 10000001):
                    if time.time() > deadline: return None
                    # Optimization: if 1/d is too small, we can't catch up?
                    # No, we just need sum to be rem.
                    # But if 1/d < rem / (k-depth), then even with max terms we can't reach?
                    # Actually, if 1/d is very small, the remaining sum needs to be made by
                    # (k-depth-1) terms.
                    # The max sum of remaining terms is bounded.
                    new_rem = rem - Fraction(1, d)
                    if new_rem < 0: continue # d too small, but d_min ensures 1/d <= rem
                    # Check if new_rem can be formed by k-depth-1 terms > d
                    # Max sum of k-depth-1 terms starting from d+1
                    if k - depth - 1 > 0:
                        max_rem = sum(Fraction(1, d+i) for i in range(1, k-depth))
                        if new_rem > max_rem:
                            # If new_rem is too big, we need a smaller d (larger 1/d)
                            # But we are iterating d upwards. So if this fails,
                            # larger d will make 1/d smaller, new_rem larger.
                            # So we can break?
                            # Yes, if new_rem > max_possible, then for larger d,
                            # 1/d decreases, new_rem increases, max_possible decreases.
                            # So it will never work.
                            break
                    
                    chosen.append(d)
                    res = dfs(new_rem, d, depth+1, chosen)
                    chosen.pop()
                    if res: return res
                return None
            
            return dfs(Fraction(1,1), 4, 0, [])
        
        # Try k=8
        res = search(8)
        if res:
            return res
        
        # Try k=9
        res = search(9)
        if res:
            return res
        
        # Fallback: return a valid k=10 solution or similar
        # Greedy for 1 with min denom 5
        rem = Fraction(1,1)
        denoms = []
        last = 4
        while rem > 0:
            inv = Fraction(1,1)/rem
            d = (inv.numerator + inv.denominator - 1) // inv.denominator
            if d < last + 1: d = last + 1
            denoms.append(d)
            rem -= Fraction(1, d)
            last = d
        return denoms
    