import numpy as np
from scipy.spatial.distance import pdist
from scipy.optimize import differential_evolution, minimize
import time
import warnings
warnings.filterwarnings('ignore')

def min_max_dist_dim2_16() -> np.ndarray:
      n = 16
      start_time = time.time()
      deadline = start_time + 590

      def get_score(points):
          dists = pdist(points, 'euclidean')
          d_min = np.min(dists)
          d_max = np.max(dists)
          if d_max < 1e-10: return 0.0
          return (d_min / d_max) ** 2

      def objective_fixed(params):
          # params: 28 values for points 2..15
          p0 = [0.0, 0.0]
          p1 = [1.0, 0.0]
          pts = np.array([p0, p1] + [list(params[i:i+2]) for i in range(0, 28, 2)])
          dists = pdist(pts, 'euclidean')
          d_min = np.min(dists)
          d_max = np.max(dists)
          if d_max < 1e-10: return 1.0 # Maximize ratio -> Minimize -ratio
          return -(d_min / d_max)
      
      def smooth_objective(params, beta=50.0):
          """
          Smooth approximation of -min(dist)/max(dist) using log-sum-exp.
          Minimizes this to maximize the ratio.
          """
          pts = params.reshape((n, 2))
          dists = pdist(pts, 'euclidean')
          
          if np.max(dists) < 1e-10:
              return 1.0
          
          # Smooth max distance: log(sum(exp(d))) ≈ max(d)
          # Smooth min distance: -log(sum(exp(-d))) ≈ min(d)
          
          # For numerical stability, use the log-sum-exp trick
          max_d = np.max(dists)
          min_d = np.min(dists)
          
          # Smooth max: max_d + log(sum(exp(d - max_d)))
          log_sum_exp_max = max_d + np.log(np.sum(np.exp(dists - max_d)))
          
          # Smooth min: -min_d - log(sum(exp(-d + min_d))) = -min_d - log(sum(exp(-(d - min_d))))
          # = -min_d - log(sum(exp(min_d - d)))
          log_sum_exp_min = -min_d - np.log(np.sum(np.exp(min_d - dists)))
          
          # We want to maximize min/max, so minimize -log(min/max) = -log(min) + log(max)
          # Or equivalently minimize log(max/min) = log(max) - log(min)
          # Using smooth versions: log_sum_exp_max - log_sum_exp_min
          
          # Actually, we want to minimize -min/max
          # Let's use: minimize (log_sum_exp_max - log_sum_exp_min) which approximates log(max/min)
          # This is proportional to -log(min/max), so minimizing it maximizes min/max
          
          return log_sum_exp_max - log_sum_exp_min

      best_points = None
      best_score = -1.0

      def update_best(pts):
          nonlocal best_points, best_score
          if pts is None or not np.all(np.isfinite(pts)): return
          s = get_score(pts)
          if s > best_score:
              best_score = s
              best_points = pts.copy()

      # 1. Global Search with Fixed Scale
      if time.time() < deadline - 120:
          bounds = [(-10, 10)] * 28
          try:
              res = differential_evolution(
                  objective_fixed, 
                  bounds=bounds, 
                  maxiter=150, 
                  popsize=12, 
                  tol=1e-5, 
                  seed=42,
                  workers=1,
                  polish=True,
                  init='sobol'
              )
              if res.success or hasattr(res, 'x'):
                  p0 = [0.0, 0.0]
                  p1 = [1.0, 0.0]
                  pts = np.array([p0, p1] + [list(res.x[i:i+2]) for i in range(0, 28, 2)])
                  update_best(pts)
          except Exception as e:
              print("DE failed:", e)

      # 2. Local Refinement using smooth surrogate, then exact objective
      if best_points is not None and time.time() < deadline - 30:
          def obj_full(params):
              pts = params.reshape((n, 2))
              dists = pdist(pts, 'euclidean')
              d_min = np.min(dists)
              d_max = np.max(dists)
              if d_max < 1e-10: return 0.0
              return -(d_min / d_max)
          
          # First refine with smooth objective (gradient-friendly)
          try:
              res_smooth = minimize(
                  lambda p: smooth_objective(p, beta=50.0), 
                  best_points.flatten(), 
                  method='L-BFGS-B', 
                  options={'maxiter': 300, 'ftol': 1e-14, 'gtol': 1e-10}
              )
              if np.all(np.isfinite(res_smooth.x)):
                  pts_smooth = res_smooth.x.reshape((n, 2))
                  s_smooth = get_score(pts_smooth)
                  if s_smooth > best_score:
                      best_score = s_smooth
                      best_points = pts_smooth.copy()
          except Exception:
              pass
          
          # Then refine with exact objective using Powell
          if best_points is not None:
              try:
                  res = minimize(obj_full, best_points.flatten(), method='Powell', options={'maxiter': 1000, 'ftol': 1e-15, 'xtol': 1e-15})
                  if np.all(np.isfinite(res.x)):
                      update_best(res.x.reshape((n, 2)))
              except Exception:
                  pass

      # 3. Fallback: Grid/Hex if DE failed or is slow
      if time.time() < deadline - 60:
          # Grid
          x = np.arange(4)
          y = np.arange(4)
          xx, yy = np.meshgrid(x, y)
          grid_pts = np.column_stack((xx.ravel(), yy.ravel()))
          # Refine grid
          def obj_full2(params):
              pts = params.reshape((n, 2))
              dists = pdist(pts, 'euclidean')
              d_min = np.min(dists)
              d_max = np.max(dists)
              if d_max < 1e-10: return 0.0
              return -(d_min / d_max)
          
          try:
              res = minimize(obj_full2, grid_pts.flatten(), method='Powell', options={'maxiter': 300, 'ftol': 1e-12})
              if np.all(np.isfinite(res.x)):
                  update_best(res.x.reshape((n, 2)))
          except Exception:
              pass
          
          # Hex
          hex_pts = np.array([
              [0, 0], [1, 0], [2, 0], [3, 0],
              [0.5, 0.866], [1.5, 0.866], [2.5, 0.866], [3.5, 0.866],
              [0, 1.732], [1, 1.732], [2, 1.732], [3, 1.732],
              [0.5, 2.598], [1.5, 2.598], [2.5, 2.598], [3.5, 2.598]
          ])
          try:
              res = minimize(obj_full2, hex_pts.flatten(), method='Powell', options={'maxiter': 300, 'ftol': 1e-12})
              if np.all(np.isfinite(res.x)):
                  update_best(res.x.reshape((n, 2)))
          except Exception:
              pass
          
          # Additional initializations: circle, spiral
          if time.time() < deadline - 30:
              # Circle
              angles = np.linspace(0, 2*np.pi, n, endpoint=False)
              circle_pts = np.column_stack([np.cos(angles), np.sin(angles)])
              try:
                  res = minimize(obj_full2, circle_pts.flatten(), method='Powell', options={'maxiter': 300, 'ftol': 1e-12})
                  if np.all(np.isfinite(res.x)):
                      update_best(res.x.reshape((n, 2)))
              except Exception:
                  pass
              
              # Spiral
              t = np.linspace(0, 4*np.pi, n)
              r = np.linspace(0.1, 1.0, n)
              spiral_pts = np.column_stack([r * np.cos(t), r * np.sin(t)])
              try:
                  res = minimize(obj_full2, spiral_pts.flatten(), method='Powell', options={'maxiter': 300, 'ftol': 1e-12})
                  if np.all(np.isfinite(res.x)):
                      update_best(res.x.reshape((n, 2)))
              except Exception:
                  pass

      # 4. High-precision final polish
      if best_points is not None and time.time() < deadline - 5:
          def obj_full_highprec(params):
              pts = params.reshape((n, 2))
              dists = pdist(pts, 'euclidean')
              d_min = np.min(dists)
              d_max = np.max(dists)
              if d_max < 1e-10: return 0.0
              return -(d_min / d_max)
          
          try:
              res = minimize(
                  obj_full_highprec, 
                  best_points.flatten(), 
                  method='Nelder-Mead', 
                  options={'maxiter': 5000, 'xatol': 1e-14, 'fatol': 1e-14, 'adaptive': True}
              )
              if np.all(np.isfinite(res.x)):
                  update_best(res.x.reshape((n, 2)))
          except Exception:
              pass
          
          # Final Powell with very tight tolerance
          if time.time() < deadline - 2:
              try:
                  res = minimize(
                      obj_full_highprec, 
                      best_points.flatten(), 
                      method='Powell', 
                      options={'maxiter': 2000, 'ftol': 1e-16, 'xtol': 1e-16}
                  )
                  if np.all(np.isfinite(res.x)):
                      update_best(res.x.reshape((n, 2)))
              except Exception:
                  pass

      if best_points is None:
          # Absolute fallback
          x = np.arange(4)
          y = np.arange(4)
          xx, yy = np.meshgrid(x, y)
          return np.column_stack((xx.ravel(), yy.ravel()))
          
      return best_points
