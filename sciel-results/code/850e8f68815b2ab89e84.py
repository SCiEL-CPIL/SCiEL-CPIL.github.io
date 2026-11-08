  positions = []
  for r in range(3):
      for c in range(4):
          x = c * 2.0 + (1.0 if r % 2 == 1 else 0.0)
          y = r * np.sqrt(3)
          positions.append((x, y))
  cx = np.mean([p[0] for p in positions])
  cy = np.mean([p[1] for p in positions])
  # Shift
  inner = []
  for (x, y) in positions:
       inner.append([x - cx, y - cy, 0.0])
  