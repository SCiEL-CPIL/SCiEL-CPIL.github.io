<<<<<<< SEARCH
                if total > 0:
                    probs = gains_float / total
                    # Sample from distribution
                    best_idx = rng.choice(num_k, p=probs)
                    best_gain = int(gains[best_idx])
                else:
                    break
=======
                if total > 0:
                    probs = gains_float / total
                    # Sample from distribution using numpy
                    best_idx = int(np.random.choice(num_k, p=probs))
                    best_gain = int(gains[best_idx])
                else:
                    break
>>>>>>> REPLACE
