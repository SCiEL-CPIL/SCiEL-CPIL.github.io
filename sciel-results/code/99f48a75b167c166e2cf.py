        for i in range(1, budget + 1):
            if time.time() > deadline:
                break
            
            u = self._halton(i, 2)
            v = self._halton(i, 3)
            
            # Transformation to uniform sphere surface
            x_0 = (2.0 / np.pi) * np.arccos(u)
            x_1 = v
            
            x = np.zeros(dim)
            x[0] = x_0
            x[1] = x_1
            x[2:] = 0.5
            
            f = problem(x)
            results.append(f)
