            if max_gain == len(unc):
                # We found a block that covers everything remaining
                # But we need to pick one. best_blocks contains all with max_gain.
                # If max_gain == len(unc), then any block in best_blocks covers all unc.
                chosen = random.choice(best_blocks)
                solution.append(chosen)
                break
