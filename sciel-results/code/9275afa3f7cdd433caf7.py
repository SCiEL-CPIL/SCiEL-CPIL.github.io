                # Add attractive values (multiples of small LCMs)
                # This is much faster than generating smooth numbers on the fly
                for _ in range(50):
                    v = random.choice(_ATTRACTION_VALUES)
                    if 1 <= v <= 10000:
                        cand_vals.add(v)
                
                # Also try values that make sums with other elements "nice"
                # e.g., v + current[j] is a multiple of 30
                for j in range(n):
                    if j == idx:
                        continue
                    target = current[j]
                    # v = 30*k - target
                    # k such that 1 <= 30k - target <= 10000
                    min_k = (target + 1 + 29) // 30
                    max_k = (target + 10000) // 30
                    if min_k <= max_k:
                        # Sample a few k
                        for _ in range(10):
                            k = random.randint(min_k, max_k)
                            v = 30 * k - target
                            if 1 <= v <= 10000:
                                cand_vals.add(v)
    