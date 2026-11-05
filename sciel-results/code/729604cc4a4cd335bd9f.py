remaining = [x for x in range(2, 129) if x not in prefix_set and (x > prefix[-1] if prefix else True)]
