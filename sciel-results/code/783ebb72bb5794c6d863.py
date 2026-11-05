    def greedy_mis_efficient(randomize_ties=False):
        remaining = set(range(m))
        # deg[i] is the number of neighbors of i in remaining
        deg = [len(conflict[i]) for i in range(m)]
        selected = []
        
        # We need to find min degree efficiently.
        # We can use a heap or just scan. Scanning is O(N).
        # Updating is O(degree).
        
        while remaining:
            # Find min degree in remaining
            min_deg = float('inf')
            candidates = []
            for idx in remaining:
                d = deg[idx]
                if d < min_deg:
                    min_deg = d
                    candidates = [idx]
                elif d == min_deg:
                    candidates.append(idx)
            
            if randomize_ties:
                best = random.choice(candidates)
            else:
                best = candidates[0]
            
            selected.append(best)
            remaining.remove(best)
            
            # Update degrees of neighbors
            for c in conflict[best]:
                if c in remaining:
                    deg[c] -= 1
        
        return selected
