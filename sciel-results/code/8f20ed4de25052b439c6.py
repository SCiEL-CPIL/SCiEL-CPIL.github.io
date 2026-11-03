                for x in candidates:
                    gain = 0
                    for rest in itertools.combinations(block, t - 1):
                        # ... mask computation ...
