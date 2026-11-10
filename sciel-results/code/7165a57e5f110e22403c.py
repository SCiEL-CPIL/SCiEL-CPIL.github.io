                if mode == 1 and len(candidates) > 1:
                    top_k = min(10, len(candidates))
                    w = int(rng.choice(candidates[:top_k]))
                elif mode == 0:
                    w = int(candidates[0])
                else:
                    w = int(rng.choice(candidates))
