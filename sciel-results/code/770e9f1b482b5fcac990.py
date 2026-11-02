    def construct():
        angles = np.linspace(0, 2*np.pi, N, endpoint=False)
        pts = np.column_stack([np.cos(angles), np.sin(angles)])
        
        current_score = score(pts)
        best_pts = pts.copy()
        best_score = current_score
        
        # Time limit 10 seconds to be safe
        start_time = time.time()
        temperature = 0.1
        step_size = 0.05
        
        while time.time() - start_time < 10:
            # Perturb
            new_pts = pts + np.random.normal(0, step_size, size=pts.shape)
            new_score = score(new_pts)
            
            if new_score > current_score:
                pts = new_pts
                current_score = new_score
                if current_score > best_score:
                    best_score = current_score
                    best_pts = pts.copy()
            else:
                # Simulated annealing acceptance
                if np.random.random() < np.exp((new_score - current_score) / temperature):
                    pts = new_pts
                    current_score = new_score
                    
            # Cool down
            temperature *= 0.99
            if temperature < 0.001:
                temperature = 0.001
                step_size *= 0.9
                if step_size < 0.001:
                    break
                    
        return best_pts
    