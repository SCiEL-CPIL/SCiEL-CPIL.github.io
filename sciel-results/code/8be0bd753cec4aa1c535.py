            remaining = target_count - len(selected)
            if remaining > 0:
                unselected_indices = np.where(unselected_mask)[0]
                unselected_scores = score[unselected_indices]
                top_add = np.argsort(-unselected_scores)[:remaining]
                for idx in top_add:
                    selected.add(int(unselected_indices[idx]))
