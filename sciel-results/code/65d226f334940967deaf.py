        # Try a few random shuffles if the first one fails
        for attempt in range(5):
            random.shuffle(candidates)
            if dfs(set(), 0, 0):
                rows = [[0] + list(marks) for marks in selected]
                return rows
