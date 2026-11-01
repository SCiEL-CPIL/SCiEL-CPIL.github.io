    def greedy_build(seed=None):
        if seed is not None:
            random.seed(seed)

        covered = np.zeros(total, dtype=np.uint8)
        code = []
        uncovered = total

        while uncovered > 0:
            new_covs = np.dot(ball_masks, 1 - covered)
            max_cov = np.max(new_covs)
            if max_cov == 0:
                break
            
            threshold = max_cov * 0.9
            candidates = np.where(new_covs >= threshold)[0]
            
            if len(candidates) == 0:
                break
                
            best_c = candidates[random.randint(0, len(candidates) - 1)]
            
            code.append(best_c)
            covered[balls[best_c]] = 1
            uncovered = np.count_nonzero(covered == 0)

        return code
