    def iterated_local_search(initial_solution, max_iters=50):
        current = list(initial_solution)
        best = list(current)
        best_size = len(current)
        
        rng = np.random.RandomState(42)
        
        for _ in range(max_iters):
            if time.time() > deadline - 5.0:
                break
                
            # Recompute coverage
            coverage = np.zeros(n_triples, dtype=np.int32)
            for bi in current:
                coverage[block_triples[bi]] += 1
                
            # Perturb
            if len(current) < 3:
                continue
                
            # Remove 2 random blocks
            idxs = rng.choice(len(current), size=2, replace=False)
            for i in sorted(idxs, reverse=True):
                bi = current[i]
                coverage[block_triples[bi]] -= 1
                current.pop(i)
                
            # Repair
            uncovered = coverage == 0
            while np.any(uncovered):
                if time.time() > deadline - 2.0:
                    break
                mask = uncovered[block_triples]
                gains = np.sum(mask, axis=1)
                best_idx = np.argmax(gains)
                if gains[best_idx] == 0:
                    break
                
                current.append(best_idx)
                coverage[block_triples[best_idx]] += 1
                uncovered[block_triples[best_idx]] = False
                
            # Polish
            current = local_search(current, max_iterations=10)
            
            # Update Best
            if len(current) < best_size:
                best = list(current)
                best_size = len(current)
            
        return best
