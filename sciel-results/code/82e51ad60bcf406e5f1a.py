        # Perturbation
        if len(g.edges) > 5:
            for _ in range(3): # Try a few perturbations
                if time.time() > deadline:
                    break
                # Remove 2 random edges
                removed = []
                edges_to_remove = random.sample(g.edges, 2)
                for e in edges_to_remove:
                    u, v = e
                    if g.remove_edge(u, v):
                        removed.append((u, v))
                
                # Run local search
                g.local_search(deadline)
                
                # If we didn't improve or get stuck, restore? 
                # Actually, local search will try to add back. 
                # If it fails, we might be worse. 
                # We keep the best graph found so far.
                
                # Try to restore if we got worse?
                # No, just let local search work on the perturbed state.
                # The global best is tracked.
